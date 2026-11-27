r"""Confidence-gated looping router (Pool 5 winning architecture: single-head Top-6).

Frozen seeds in a pool directory (registry.json) share one tokenizer and socket.
A context-only base router picks a sparse active set per token and mixes their
next-token distributions. A confidence-gated loop then takes 1..3 passes: if
pass-r confidence < tau it adds more seeds (top step_k by updated score) and
re-mixes every seed used so far.

Winner (Pool 5): --mode single --init-k 6 --step-k 3  (macro BPB 0.8293, 17.2 tok/s)

    python pool5/eval/pool5_loop.py base --dir pool5 --mode single --init-k 6 --step-k 3
    python pool5/eval/pool5_loop.py loop --dir pool5 --minutes 15 --mix-json path/to/mix.json
    python pool5/eval/pool5_loop.py chat --dir pool5 --tools --prompt "What is 17 * 19?"

Adding or removing seeds changes the router key set: re-run `base` then `loop`.
`--mix-json` is a list of {name, path, weight} (or name: [weight, path] objects).
Missing files are skipped. If nothing is left, training falls back to data/train.bin.
"""
import sys, os, json, time, csv, math, argparse, re
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import seed_v4 as S, seed_pool as P
MAXP = 3


def load_mix_json(path, log=print):
    """Return {name: (weight, path)} for files that exist. Accepts a list or a dict."""
    extra = {}
    if not path:
        return extra
    p = Path(path)
    if not p.is_file():
        log(f"mix-json not found: {p}")
        return extra
    raw = json.loads(p.read_text(encoding="utf-8"))
    rows = raw if isinstance(raw, list) else [{"name": k, **(v if isinstance(v, dict) else {"weight": v[0], "path": v[1]})} for k, v in raw.items()]
    for row in rows:
        name = row.get("name") or row.get("id")
        src = row.get("path") or row.get("file")
        w = float(row.get("weight", 1.0))
        if not name or not src:
            continue
        fp = Path(src)
        if not fp.is_file():
            log(f"mix skip {name}: missing {fp}")
            continue
        extra[name] = (w, str(fp))
    log("mix: " + (", ".join(f"{n}:{w}" for n, (w, _) in extra.items()) if extra else "(none; using data/train.bin)"))
    return extra

# ---------------------------------- base context routers ----------------------------------
class CtxRouter(nn.Module):
    """Single-head context-only router (exactly pool3/pool4's proven CtxRouter)."""
    def __init__(self, names, vocab=4096, d_tok=64, d_ctx=192, d_key=128):
        super().__init__()
        self.mode="single"; self.d_key=d_key
        self.emb = nn.Embedding(vocab, d_tok); self.gru = nn.GRU(d_tok, d_ctx, batch_first=True)
        self.wc = nn.Linear(d_ctx + d_tok, d_key); self.v = nn.Linear(d_key, 1, bias=False)
        self.keys = nn.ParameterDict({n: nn.Parameter(torch.randn(d_key)*0.02) for n in names})
        nn.init.normal_(self.emb.weight, std=0.1)
    def ctx(self, idx):
        e = self.emb(idx); c,_ = self.gru(e); return self.wc(torch.cat([c,e],-1))   # B,T,d_key
    def scores(self, idx, names, c=None):
        c = self.ctx(idx) if c is None else c
        return torch.stack([self.v(torch.tanh(c + self.keys[n])).squeeze(-1) for n in names], -1)  # B,T,N

class MHCtxRouter(nn.Module):
    """Multi-head context-only router: H heads, per-seed per-head keys; pass-1 active set = union of each head's top-k."""
    def __init__(self, names, heads=3, vocab=4096, d_tok=64, d_ctx=192, d_key=128):
        super().__init__()
        self.mode="multi"; self.H=heads; self.d_key=d_key
        self.emb = nn.Embedding(vocab, d_tok); self.gru = nn.GRU(d_tok, d_ctx, batch_first=True)
        self.wc = nn.Linear(d_ctx + d_tok, d_key)
        self.v = nn.ModuleList([nn.Linear(d_key, 1, bias=False) for _ in range(heads)])
        self.keys = nn.ParameterDict({n: nn.Parameter(torch.randn(heads, d_key)*0.02) for n in names})
        nn.init.normal_(self.emb.weight, std=0.1)
    def ctx(self, idx):
        e = self.emb(idx); c,_ = self.gru(e); return self.wc(torch.cat([c,e],-1))
    def head_scores(self, idx, names, c=None):
        c = self.ctx(idx) if c is None else c
        hs = []
        for h in range(self.H):
            hs.append(torch.stack([self.v[h](torch.tanh(c + self.keys[n][h])).squeeze(-1) for n in names], -1))
        return torch.stack(hs, -1)   # B,T,N,H
    def scores(self, idx, names, c=None):
        return self.head_scores(idx, names, c).mean(-1)   # combined scalar score B,T,N

def topk_mask(sc, k):
    k = min(k, sc.size(-1)); ix = sc.topk(k, -1).indices
    return torch.zeros_like(sc, dtype=torch.bool).scatter(-1, ix, True)

def initial_select(router, idx, names, init_k, c=None):
    """Pass-1 active mask. single: top init_k of scores. multi: union of each head's top init_k."""
    if router.mode == "single":
        return topk_mask(router.scores(idx, names, c), init_k)
    hs = router.head_scores(idx, names, c)                 # B,T,N,H
    m = torch.zeros(hs.shape[:-1], dtype=torch.bool, device=hs.device)
    for h in range(router.H):
        m = m | topk_mask(hs[..., h], init_k)
    return m

# ---------------------------------- loop head ----------------------------------
class RLoop(nn.Module):
    def __init__(self, N, d_key=128, d_emb=384, h=256):
        super().__init__(); self.pe = nn.Embedding(MAXP, 32); din = d_key + d_emb + 2 + 32
        self.delta = nn.Sequential(nn.Linear(din, h), nn.GELU(), nn.Linear(h, N)); nn.init.zeros_(self.delta[2].weight); nn.init.zeros_(self.delta[2].bias)
        self.conf  = nn.Sequential(nn.Linear(din, h), nn.GELU(), nn.Linear(h, 1)); nn.init.constant_(self.conf[2].bias, 1.0)
    def feats(self, c, plog, E, r):
        p = plog.exp(); e = p @ E; ent = -(p*plog).sum(-1, keepdim=True)/math.log(p.size(-1)); mx = p.amax(-1, keepdim=True)
        return torch.cat([c, e, ent, mx, self.pe.weight[r].expand(*c.shape[:-1], 32)], -1)

def add_select(sc, used, step_k):
    s = sc.masked_fill(used, -float("inf")); ix = s.topk(min(step_k, sc.size(-1)), -1).indices
    return used | torch.zeros_like(used).scatter(-1, ix, True)
def mixed(lps, sc, used):   # lps (...,V,N) log-probs, mask used (...,N) -> (...,V)
    lw = torch.log_softmax(sc.masked_fill(~used, -float("inf")), -1); return torch.logsumexp(lps + lw.unsqueeze(-2), -1)

# ---------------------------------- pool wrapper ----------------------------------
class P5:
    def __init__(self, dir, device="cuda", extra=None):
        self.dir = Path(dir); self.dev = device
        self.extra = extra or {}
        self.pool = P.load_pool(dir, device)       # seeds + (unused dense router, if present)
        self.names = self.pool.names; self.seeds = self.pool.seeds
        self.ctx = self.pool.ctx
        self.E = next(iter(self.seeds.values())).emb.weight.detach().float()
        self.cfg = dict(mode="single", heads=3, init_k=6, step_k=3)
        self.router = None; self.rl = None
    def build_router(self):
        if self.cfg["mode"] == "single": self.router = CtxRouter(self.names).to(self.dev)
        else: self.router = MHCtxRouter(self.names, self.cfg["heads"]).to(self.dev)
    def save_base(self):
        torch.save(dict(rt=self.router.state_dict(), cfg=self.cfg, names=self.names), self.dir/"router_base.pt")
    def load_base(self):
        st = torch.load(self.dir/"router_base.pt", map_location=self.dev, weights_only=False)
        self.cfg = st["cfg"]; assert st["names"] == self.names; self.build_router()
        self.router.load_state_dict(st["rt"]); self.router.to(self.dev).eval()
    def save_loop(self):
        torch.save(dict(rl=self.rl.state_dict(), cfg=self.cfg), self.dir/"router_loop.pt")
    def load_loop(self):
        self.rl = RLoop(len(self.names)).to(self.dev)
        self.rl.load_state_dict(torch.load(self.dir/"router_loop.pt", map_location=self.dev, weights_only=False)["rl"]); self.rl.eval()
    # seed target log-probs (B,T,N) of true next token y
    @torch.no_grad()
    def seed_ly(self, idx, y):
        return torch.stack([F.log_softmax(P.seed_forward(self.seeds[n], idx)[0], -1).gather(-1, y.unsqueeze(-1)).squeeze(-1) for n in self.names], -1)
    @torch.no_grad()
    def seed_lps(self, idx):
        return torch.stack([F.log_softmax(P.seed_forward(self.seeds[n], idx)[0], -1) for n in self.names], -1)   # B,T,V,N

# ---------------------------------- base router training ----------------------------------
def train_base(p5, max_steps, batch, T, lr, eval_every, log):
    tok = S.find_tokenizer(next(iter(p5.seeds.values())))
    data = P.MixData(tok, ts_frac=0.1 if p5.extra else 1.0, extra=p5.extra or None); rng = np.random.default_rng(0); vr = np.random.default_rng(777)
    val = {"ts":[data.batch(8,T,vr,"ts") for _ in range(4)], "tool":[data.batch(8,T,vr,"tool") for _ in range(8)]}
    for s in p5.seeds.values():
        for q in s.parameters(): q.requires_grad_(False)
    params = list(p5.router.parameters()); opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    ik, sk = p5.cfg["init_k"], p5.cfg["step_k"]; t0=time.time(); run=[]; hist=[]
    mp = p5.dir/"metrics_base.csv"
    with open(mp,"w",newline="") as fh: csv.writer(fh).writerow(["step","min","train","val_ts","val_tool","mean_active"]+[f"w_{n}" for n in p5.names])
    def losses(idx, y):
        c = p5.router.ctx(idx); sc = p5.router.scores(idx, p5.names, c); sel = initial_select(p5.router, idx, p5.names, ik, c)
        ly = p5.seed_ly(idx, y)
        lw_sel = torch.log_softmax(sc.masked_fill(~sel,-float("inf")),-1); lw_full = torch.log_softmax(sc,-1)
        nsel = -torch.logsumexp(ly+lw_sel,-1).mean(); nful = -torch.logsumexp(ly+lw_full,-1).mean()
        return nsel, nful, sel.float().sum(-1).mean(), lw_full.detach().exp().mean(dim=(0,1))
    def ev(bs):
        tot=0.0; ws=[]; act=[]
        with torch.no_grad():
            for x,y in bs:
                ns,_,a,w = losses(x.to(p5.dev), y.to(p5.dev)); tot+=float(ns); ws.append(w); act.append(float(a))
        return tot/len(bs), torch.stack(ws).mean(0).tolist(), float(np.mean(act))
    for step in range(1, max_steps+1):
        m = min(1.0, step/100)*(1.0 if step<max_steps*0.8 else max(0.1,(max_steps-step)/(max_steps*0.2)))
        for g in opt.param_groups: g["lr"]=lr*m
        x,y = data.batch(batch,T,rng); x,y = x.to(p5.dev), y.to(p5.dev)
        ns,nf,_,_ = losses(x,y); loss = ns + 0.5*nf
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params,1.0); opt.step(); run.append(float(loss))
        if step%eval_every==0 or step==1:
            vt,wt,at = ev(val["ts"]); vo,wo,ao = ev(val["tool"]); vm=0.4*vt+0.6*vo; hist.append((step,vm))
            log(f"[base {p5.cfg['mode']}] step {step} train {np.mean(run[-eval_every:]):.4f} | val ts {vt:.4f} tool {vo:.4f} act {ao:.1f} | "
                + " ".join(f"{n}={w:.2f}" for n,w in zip(p5.names,wo)) + f" | {time.time()-t0:.0f}s vram {torch.cuda.max_memory_allocated()/2**20:.0f}MB")
            with open(mp,"a",newline="") as fh: csv.writer(fh).writerow([step,round((time.time()-t0)/60,2),round(float(np.mean(run[-eval_every:])),4),round(vt,4),round(vo,4),round(ao,2)]+[round(w,4) for w in wo])
            old=[v for s_,v in hist if s_<=step-500]
            if old and step<max_steps*0.8 and (old[-1]-vm)/old[-1]<0.003: log(f"[base] flattened at step {step}"); break
    p5.router.eval(); return dict(steps=step, seconds=round(time.time()-t0), val=hist[-1][1] if hist else None)

# ---------------------------------- loop head training ----------------------------------
def all_passes(p5, idx, lps=None):
    names=p5.names; c=p5.router.ctx(idx); sc=p5.router.scores(idx, names, c)
    if lps is None: lps = p5.seed_lps(idx)
    ik, sk = p5.cfg["init_k"], p5.cfg["step_k"]
    used = initial_select(p5.router, idx, names, ik, c); outs=[]; confs=[]
    for r in range(MAXP):
        pl = mixed(lps, sc, used); outs.append(pl)
        if r==MAXP-1: break
        f = p5.rl.feats(c, pl, p5.E, r); confs.append(torch.sigmoid(p5.rl.conf(f.detach())).squeeze(-1))
        sc = sc + p5.rl.delta(f); used = add_select(sc, used, sk)
    return outs, confs

def train_loop(p5, minutes, batch, T, lr, log):
    tok = S.find_tokenizer(next(iter(p5.seeds.values())))
    data = P.MixData(tok, ts_frac=0.1 if p5.extra else 1.0, extra=p5.extra or None); rng = np.random.default_rng(0)
    for q in p5.router.parameters(): q.requires_grad_(False)
    p5.rl = RLoop(len(p5.names)).to(p5.dev)
    opt = torch.optim.AdamW(p5.rl.parameters(), lr=lr, weight_decay=0.01); t0=time.time(); step=0; ema=None
    while time.time()-t0 < minutes*60:
        x,y = data.batch(batch,T,rng); x,y = x.to(p5.dev), y.to(p5.dev)
        outs, confs = all_passes(p5, x)
        nll = [-o.gather(-1,y.unsqueeze(-1)).squeeze(-1) for o in outs]
        bce = 0.0
        for r in range(MAXP-1):
            better = torch.stack(nll[r+1:],0).amin(0).detach(); lab = ((nll[r].detach()-better)<0.05).float()
            bce = bce + F.binary_cross_entropy(confs[r].clamp(1e-5,1-1e-5), lab)
        loss = nll[1].mean() + nll[2].mean() + bce
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(p5.rl.parameters(),1.0); opt.step(); step+=1
        v=[float(n.mean()) for n in nll]+[float(bce)]+[float(c.mean()) for c in confs]; ema = v if ema is None else [0.97*a+0.03*b for a,b in zip(ema,v)]
        if step%25==0: log(f"[loop] step {step} nll p1 {ema[0]:.4f} p2 {ema[1]:.4f} p3 {ema[2]:.4f} bce {ema[3]:.3f} conf1 {ema[4]:.2f} conf2 {ema[5]:.2f} {time.time()-t0:.0f}s")
        if step%200==0: p5.save_loop()
    p5.save_loop(); log(f"[loop] saved after {step} steps"); return dict(steps=step)

# ---------------------------------- generation (route last position, run only needed seeds) ----------------------------------
@torch.no_grad()
def step_last(p5, idx, tau, stats=None):
    names=p5.names; c=p5.router.ctx(idx)[:, -1]; sc=p5.router.scores(idx, names, c=None)[:, -1]
    used = initial_select(p5.router, idx, names, p5.cfg["init_k"])[:, -1]; cache={}
    for r in range(MAXP):
        for j in used[0].nonzero().flatten().tolist():
            if j not in cache: cache[j] = F.log_softmax(P.seed_forward(p5.seeds[names[j]], idx)[0][:, -1].float(), -1)[0]
        js = sorted(cache); lps = torch.stack([cache[j] for j in js], -1)
        lw = torch.log_softmax(sc[0, js], -1); pl = torch.logsumexp(lps+lw, -1)
        if r==MAXP-1: break
        f = p5.rl.feats(c, pl[None], p5.E, r); conf = float(torch.sigmoid(p5.rl.conf(f)))
        if conf >= tau: break
        sc = sc + p5.rl.delta(f); used = add_select(sc, used, p5.cfg["step_k"])
    if stats is not None:
        stats["passes"][r+1]+=1; stats["seed_runs"]+=len(cache)
        for j in cache: stats["routed"][names[j]] = stats["routed"].get(names[j],0)+1
    return pl   # (V,) log-probs

def load(dir, device="cuda", with_loop=True, extra=None):
    p5 = P5(dir, device, extra=extra); p5.load_base()
    if with_loop: p5.load_loop()
    return p5

# ---------------------------------- generation / chat ----------------------------------
@torch.no_grad()
def generate(p5, tok, text, n=200, tau=0.9, temp=0.0, stops=(), stats=None):
    """Greedy if temp<=0, otherwise sample. Returns the newly generated string."""
    ids = list(tok.encode(text).ids)
    stop_frag = list(stops) if stops else []
    dev = p5.dev
    for _ in range(n):
        idx = torch.tensor([ids], device=dev)
        pl = step_last(p5, idx, tau, stats)
        if temp and temp > 0:
            nxt = int(torch.multinomial(torch.softmax(pl / temp, -1), 1))
        else:
            nxt = int(pl.argmax())
        ids.append(nxt)
        dec = tok.decode(ids)
        if any(dec.endswith(s) for s in stop_frag):
            break
        if tok.token_to_id("<|end|>") is not None and nxt == tok.token_to_id("<|end|>"):
            break
        if tok.token_to_id("<eos>") is not None and nxt == tok.token_to_id("<eos>"):
            break
    return tok.decode(ids[len(tok.encode(text).ids):])


def chat_once(p5, tok, prompt, n=200, tau=0.9, temp=0.0, tools=False):
    if not tools:
        return generate(p5, tok, prompt, n=n, tau=tau, temp=temp)
    try:
        from tools.runtime import SPECS, run_tool
    except Exception:
        import comparison.tiny100m.chat100m as C
        SPECS, run_tool = C.SPECS, C.run_tool
    ws = os.environ.get("SEED_WORKSPACE", str(REPO / "workspace"))
    Path(ws).mkdir(parents=True, exist_ok=True)
    text = "<|system|>You are a helpful assistant. You can call these tools:\n" + "\n".join(json.dumps(v) for v in SPECS.values()) + "<|end|>\n"
    text += f"<|user|>{prompt}<|end|>\n<|assistant|>"
    full = ""
    for _ in range(3):
        new = generate(p5, tok, text, n=n, tau=tau, temp=temp, stops=("<|end|>", "</tool_call>"))
        text += new; full += new
        mt = re.search(r"<tool_call>(.*?)</tool_call>$", new, re.S)
        if not mt:
            break
        res = run_tool(mt.group(1), ws)
        piece = f"<tool_result>{res}</tool_result>"
        text += piece; full += piece
    return full


# ---------------------------------- CLI ----------------------------------
def main():
    ap = argparse.ArgumentParser(description="Confidence-gated looping router (Pool 5 Top-6 winner)")
    ap.add_argument("cmd", choices=["base", "loop", "chat", "list"])
    ap.add_argument("--dir", default="pool5", help="pool directory with registry.json (and router_*.pt for loop/chat)")
    ap.add_argument("--mode", default="single", choices=["single", "multi"])
    ap.add_argument("--heads", type=int, default=3)
    ap.add_argument("--init-k", type=int, default=6, help="pass-1 active set size (winner uses 6)")
    ap.add_argument("--step-k", type=int, default=3, help="extra seeds added on a low-confidence pass")
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--minutes", type=float, default=15.0)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--T", type=int, default=384)
    ap.add_argument("--mix-json", default=None, help="optional training mix for base/loop (JSON list of {name,path,weight})")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tau", type=float, default=0.9, help="loop confidence threshold for chat")
    ap.add_argument("--temp", type=float, default=0.0, help="0 = greedy (the reported tok/s and coding evals)")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--tools", action="store_true")
    a = ap.parse_args()
    d = Path(a.dir)
    if not d.is_absolute():
        d = (Path.cwd() / d).resolve()
    def log(s):
        print(s, flush=True)
        if a.cmd in ("base", "loop"):
            lg.write(time.strftime("[%H:%M:%S] ") + s + "\n"); lg.flush()
    extra = load_mix_json(a.mix_json, log) if a.mix_json else {}
    p5 = P5(d, a.device, extra=extra)
    if a.cmd == "list":
        print(f"pool {d}")
        print("seeds:", ", ".join(p5.names))
        print(f"cfg default {p5.cfg}")
        print("router_base", (d / "router_base.pt").is_file(), "router_loop", (d / "router_loop.pt").is_file())
        return
    lg = open(d / f"{a.cmd}.log", "a", encoding="utf-8") if a.cmd in ("base", "loop") else None
    if a.cmd == "base":
        p5.cfg = dict(mode=a.mode, heads=a.heads, init_k=a.init_k, step_k=a.step_k); p5.build_router()
        log(f"train base {a.mode} heads={a.heads} init_k={a.init_k} step_k={a.step_k} over {p5.names}")
        r = train_base(p5, a.max_steps, a.batch, a.T, 2e-3, 100, log); p5.save_base(); log("base done " + json.dumps(r))
    elif a.cmd == "loop":
        p5.load_base(); log(f"train loop cfg={p5.cfg}")
        r = train_loop(p5, a.minutes, max(4, a.batch // 2), 256, 1e-3, log); log("loop done " + json.dumps(r))
    elif a.cmd == "chat":
        p5.load_base(); p5.load_loop()
        tok = S.find_tokenizer(next(iter(p5.seeds.values())))
        if a.prompt:
            print(chat_once(p5, tok, a.prompt, n=a.n, tau=a.tau, temp=a.temp, tools=a.tools)); return
        while True:
            try:
                q = input("> ")
            except (EOFError, KeyboardInterrupt):
                break
            if q.strip() in ("/quit", "/q"):
                break
            print(chat_once(p5, tok, q, n=a.n, tau=a.tau, temp=a.temp, tools=a.tools))

if __name__ == "__main__":
    main()
