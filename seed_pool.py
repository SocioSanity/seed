"""Seed Pool v1: a model made of whole, frozen Seeds ("organs", each with its own internal organs) and a small trained router.

Every active seed runs on every token. Per token, the router gives each seed a weight g_i (softmax over the ACTIVE seeds)
and the pool predicts with a mixture of the seeds' distributions:  log p(x) = logsumexp_i( log g_i + log_softmax(logits_i) ).
This works because every socket seed shares the tokenizer (seed-bpe4096-v1) and the frozen anchor readout.

Router (variable number of seeds): score_i = v . tanh( Wc ctx_t + Wh h_i,t + We [entropy_i,t, maxlogp_i,t] + key_i )
  ctx_t   = GRU over a tiny token embedding of the recent tokens (causal)
  h_i,t   = seed i's own final normalized hidden state (all seeds write in the same anchor space)
  key_i   = a learned per-seed descriptor. A new seed only needs its key calibrated (or a short router fine-tune).
Pool management: registry (name, description, path, size), an active set under a weights budget (default 2 GB),
hot add / remove at runtime (others untouched), per-seed usefulness = running average router weight (evicts the least
useful seed when the budget is tight), optional prefetch of candidates to CPU RAM.

CLI (from the Seed folder):
  python seed_pool.py init                      # registry with run3 + run4 in D:\\seed_v4\\pool1
  python seed_pool.py train [--max-steps 4000]  # router only; seeds frozen; stops when the val curve flattens
  python seed_pool.py chat [--prompt "..."] [--tools]
  python seed_pool.py list | add NAME PATH [--desc ...] | remove NAME | calibrate NAME [--steps 200]
"""
import argparse, json, math, os, sys, time, csv, random
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import seed_v4 as S

POOL_DIR = Path(r"D:\seed_v4\pool1")
SEP = "<|endoftext|>"


# ----------------------------------------------------------------------------- one seed: logits + final hidden state
def _seed_pass(m, idx):
    x = m.emb(idx)
    for layer in m.layers:
        x, _, _ = layer(x, m.fc, None, False)
    hn = F.rms_norm(x, (m.d,), m.nf)
    W = m.emb.weight if m.tie else m.head
    lg = F.linear(hn, W.to(hn.dtype)).float()
    if m.softcap:
        lg = m.softcap * torch.tanh(lg / m.softcap)
    return lg, hn


def seed_forward(m, idx):
    """Logits and final hidden for every position; sequences longer than the seed's context use sliding windows (stride ctx/2)."""
    T, W = idx.size(1), m.ctx
    if T <= W:
        return _seed_pass(m, idx)
    St = W // 2
    l0, h0 = _seed_pass(m, idx[:, :W]); ls, hs = [l0], [h0]; a = W
    while a < T:
        b = min(T, a + St); lo = b - W
        l_, h_ = _seed_pass(m, idx[:, lo:b]); ls.append(l_[:, a - lo:]); hs.append(h_[:, a - lo:]); a = b
    return torch.cat(ls, 1), torch.cat(hs, 1)


# ----------------------------------------------------------------------------- router
class Router(nn.Module):
    def __init__(self, vocab=4096, d_tok=64, d_ctx=128, d_seed=384, d_key=128):
        super().__init__()
        self.d_key = d_key
        self.emb = nn.Embedding(vocab, d_tok)
        self.gru = nn.GRU(d_tok, d_ctx, batch_first=True)
        self.wc = nn.Linear(d_ctx, d_key)
        self.wh = nn.Linear(d_seed, d_key, bias=False)
        self.we = nn.Linear(2, d_key, bias=False)
        self.v = nn.Linear(d_key, 1, bias=False)
        self.keys = nn.ParameterDict()
        nn.init.normal_(self.emb.weight, std=0.1)

    def add_key(self, name, init=None):
        if name not in self.keys:
            self.keys[name] = nn.Parameter(init.clone().detach() if init is not None else torch.zeros(self.d_key, device=self.v.weight.device))

    def forward(self, idx, feats, names):
        c, _ = self.gru(self.emb(idx))
        c = self.wc(c)
        sc = []
        for (h, e, mx), n in zip(feats, names):
            z = torch.tanh(c + self.wh(h) + self.we(torch.stack([e / 8.0, mx / 8.0], -1)) + self.keys[n])
            sc.append(self.v(z).squeeze(-1))
        return torch.log_softmax(torch.stack(sc, -1), -1)     # B, T, N (log weights over active seeds)


# ----------------------------------------------------------------------------- pool
class SeedPool(nn.Module):
    def __init__(self, budget_mb=2048.0, device="cuda"):
        super().__init__()
        self.seeds = nn.ModuleDict()
        self.router = Router()
        self.meta, self.stats, self.prefetched = {}, {}, {}
        self.budget_mb, self.dev = float(budget_mb), device
        self.decode_prior = None
        self.ctx, self.tokenizer, self.anchor = 512, None, None
        self.stat_decay = 0.98
        self.log = print

    @property
    def names(self):
        return list(self.seeds.keys())

    def seed_mb(self, m):
        return sum(p.numel() for p in m.parameters()) * 4 / 2**20

    def weights_mb(self):
        return sum(p.numel() for p in self.parameters()) * 4 / 2**20

    def _load(self, path):
        m, info = S.load_model_any(path, use_ema=True, device="cpu")
        for p in m.parameters():
            p.requires_grad_(False)
        return m.eval(), info

    def prefetch(self, name, path):
        """Load a candidate seed into CPU RAM (not active, costs no router time) so a later add is instant."""
        if name not in self.prefetched and name not in self.seeds:
            self.prefetched[name] = self._load(path)
        return name

    def add_seed(self, name, path, description="", key_init="mean"):
        if name in self.seeds:
            return dict(added=False, reason="already active")
        m, info = self.prefetched.pop(name) if name in self.prefetched else self._load(path)
        if not m.socket:
            raise ValueError(f"{name}: not a seed-socket model")
        if self.tokenizer is not None and m.tokenizer.get("sha256") != self.tokenizer:
            raise ValueError(f"{name}: tokenizer differs from the pool's ({m.tokenizer.get('id')})")
        if self.anchor is not None and m.anchor.get("sha256") != self.anchor:
            raise ValueError(f"{name}: socket anchor differs from the pool's")
        need = self.seed_mb(m); evicted = []
        while self.names and sum(self.seed_mb(s) for s in self.seeds.values()) + need > self.budget_mb:
            low = min(self.names, key=lambda n: self.stats.get(n, 0.0))
            evicted.append(low); self.remove_seed(low, reason="budget")
        self.tokenizer, self.anchor = m.tokenizer.get("sha256"), m.anchor.get("sha256")
        self.seeds[name] = m.to(self.dev)
        if name not in self.router.keys:
            ks = [self.router.keys[n].detach() for n in self.router.keys if n in self.seeds and n != name]
            self.router.add_key(name, torch.stack(ks).mean(0) if (ks and key_init == "mean") else None)
        self.meta[name] = dict(name=name, description=description, path=str(path), params=sum(p.numel() for p in m.parameters()),
                               weights_mb=round(need, 1), shape=m.shape_str(), ctx=m.ctx, step=info.get("step"), active=True)
        self.stats.setdefault(name, 1.0 / max(1, len(self.names)))
        self.ctx = max(s.ctx for s in self.seeds.values())
        if self.decode_prior is None:
            self.decode_prior = m.__dict__.get("decode_prior")
        self.log(f"[pool] added {name} ({m.shape_str()}, {need:.1f} MB); active {self.names}" + (f"; evicted {evicted}" if evicted else ""))
        return dict(added=True, evicted=evicted)

    def remove_seed(self, name, reason="manual"):
        """Hot remove: the seed is unloaded; its router key stays, so re-adding it later needs no calibration."""
        if name not in self.seeds:
            return False
        del self.seeds[name]
        if name in self.meta:
            self.meta[name]["active"] = False
        self.ctx = max([s.ctx for s in self.seeds.values()] or [512])
        if self.dev == "cuda":
            torch.cuda.empty_cache()
        self.log(f"[pool] removed {name} ({reason}); active {self.names}")
        return True

    def forward(self, idx, return_weights=False, update_stats=None):
        names = self.names
        lps, feats = [], []
        with torch.no_grad():
            for n in names:
                lg, h = seed_forward(self.seeds[n], idx)
                lp = F.log_softmax(lg, -1)
                ent = -(lp.exp() * lp).sum(-1); mx = lp.max(-1).values
                lps.append(lp); feats.append((h, ent, mx))
        lw = self.router(idx, feats, names)                                       # B,T,N
        mix = torch.logsumexp(torch.stack(lps, -1) + lw.unsqueeze(-2), -1)        # B,T,V log-probs
        if update_stats if update_stats is not None else not self.training:
            w = lw.detach().exp().mean(dim=(0, 1)).tolist()
            for n, wi in zip(names, w):
                self.stats[n] = self.stat_decay * self.stats.get(n, wi) + (1 - self.stat_decay) * wi
        return (mix, lw.exp()) if return_weights else mix

    @torch.no_grad()
    def generate(self, ids, n, temp=0.7, top_k=40, stops=(), tok=None, eos=3):
        out = []
        for _ in range(n):
            x = torch.tensor([(ids + out)[-self.ctx:]], device=self.dev)
            lg = self(x)[0, -1]
            if self.decode_prior is not None:
                lg = lg + S.prior_bias(self.decode_prior, (ids + out)[-1], lg.numel(), lg.device)
            lg = lg / max(temp, 1e-5)
            if top_k:
                v, _ = torch.topk(lg, top_k); lg[lg < v[-1]] = -float("inf")
            nxt = int(torch.multinomial(F.softmax(lg, -1), 1)); out.append(nxt)
            if nxt == eos or (tok is not None and stops and any(tok.decode(out[-8:]).endswith(s_) for s_ in stops)):
                break
        return out

    # -- persistence
    def save(self, d=POOL_DIR):
        d = Path(d); d.mkdir(parents=True, exist_ok=True)
        torch.save(dict(router=self.router.state_dict(), stats=self.stats, budget_mb=self.budget_mb), d / "router.pt")
        reg = json.loads((d / "registry.json").read_text()) if (d / "registry.json").exists() else {}
        reg.setdefault("seeds", {})
        for n, mt in self.meta.items():
            reg["seeds"][n] = dict(mt, active=n in self.seeds)
        reg["budget_mb"] = self.budget_mb
        (d / "registry.json").write_text(json.dumps(reg, indent=1))


def load_pool(d=POOL_DIR, device="cuda", budget_mb=None, only=None):
    d = Path(d)
    reg = json.loads((d / "registry.json").read_text())
    pool = SeedPool(budget_mb or reg.get("budget_mb", 2048), device).to(device)
    rp = d / "router.pt"
    st = torch.load(rp, map_location="cpu", weights_only=False) if rp.exists() else None
    if st:
        for k in st["router"]:
            if k.startswith("keys."):
                pool.router.add_key(k[5:], torch.zeros(pool.router.d_key))
        pool.router.load_state_dict(st["router"]); pool.router.to(device)
        pool.stats.update(st.get("stats", {}))
    for n, s in reg["seeds"].items():
        if (only is None and s.get("active", True)) or (only is not None and n in only):
            pool.add_seed(n, s["path"], s.get("description", ""))
    pool.meta.update({n: s for n, s in reg["seeds"].items() if n not in pool.meta})
    return pool


# ----------------------------------------------------------------------------- data (router training: training regions only)
TOOL_SRC = {"wiki_tool": 0.40, "glaive": 0.153, "instruct": 0.153, "calc": 0.118, "tools_misc": 0.106, "gsm8k": 0.071}


class MixData:
    """TinyStories (train.bin) + run4's tool/reasoning files, only the first 99% of each file (the last 1% is the test tail)."""
    def __init__(self, tok, ts_frac=0.4, data_dir=HERE / "data", run4_dir=r"D:\seed_v4\run4_data"):
        self.tok, self.ts_frac, self.eos = tok, ts_frac, tok.token_to_id("<eos>")
        self.ts = np.memmap(Path(data_dir) / "train.bin", dtype=np.uint16, mode="r")
        self.src = []
        for n, w in TOOL_SRC.items():
            p = Path(run4_dir) / f"{n}.txt"; size = p.stat().st_size
            self.src.append(dict(name=n, w=w, f=open(p, "rb"), cut=int(size * 0.99)))
        self.p = np.array([s["w"] for s in self.src]); self.p /= self.p.sum()

    def tool_row(self, s, T, rng):
        ids = []
        while len(ids) < T + 1:
            off = int(rng.integers(0, s["cut"] - 8192)); s["f"].seek(off); raw = s["f"].read(4096)
            k = raw.find(SEP.encode()); raw = raw[k + len(SEP):] if k >= 0 else raw
            for j, pc in enumerate(raw.decode("utf-8", errors="ignore").split(SEP)):
                if j: ids.append(self.eos)
                if pc: ids += self.tok.encode(pc).ids
        return ids[:T + 1]

    def batch(self, B, T, rng, kind=None):
        rows = []
        for _ in range(B):
            use_ts = (rng.random() < self.ts_frac) if kind is None else kind == "ts"
            if use_ts:
                i = int(rng.integers(0, len(self.ts) - T - 2)); rows.append(self.ts[i:i + T + 1].astype(np.int64).tolist())
            else:
                rows.append(self.tool_row(self.src[int(rng.choice(len(self.src), p=self.p))], T, rng))
        a = torch.tensor(rows, dtype=torch.long)
        return a[:, :-1], a[:, 1:]


def eval_pool(pool, batches, dev):
    pool.eval(); tot, ws = 0.0, []
    with torch.no_grad():
        for x, y in batches:
            mix, w = pool(x.to(dev), return_weights=True, update_stats=False)
            tot += float(F.nll_loss(mix.reshape(-1, mix.size(-1)), y.to(dev).reshape(-1))); ws.append(w.mean(dim=(0, 1)))
    pool.train()
    return tot / len(batches), torch.stack(ws).mean(0).tolist()


def train_router(pool, out=POOL_DIR, max_steps=4000, batch=16, T=512, lr=2e-3, only_keys=None, eval_every=100, patience_rel=0.003, window=500, log=print):
    """Router-only training (seeds frozen). only_keys=[name] = key calibration for a newly added seed."""
    tok = S.find_tokenizer(next(iter(pool.seeds.values())))
    data = MixData(tok); rng = np.random.default_rng(0); vr = np.random.default_rng(777)
    val = {"ts": [data.batch(8, T, vr, "ts") for _ in range(4)], "tool": [data.batch(8, T, vr, "tool") for _ in range(8)]}
    params = [pool.router.keys[n] for n in only_keys] if only_keys else list(pool.router.parameters())
    for p in pool.router.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0 if only_keys else 0.01)
    hist, t0, mp = [], time.time(), Path(out) / ("metrics.csv" if not only_keys else f"calib_{'_'.join(only_keys)}.csv")
    with open(mp, "w", newline="") as fh:
        csv.writer(fh).writerow(["step", "min", "train", "val_ts", "val_tool"] + [f"w_ts_{n}" for n in pool.names] + [f"w_tool_{n}" for n in pool.names])
    pool.train(); run, step = [], 0
    for step in range(1, max_steps + 1):
        m = min(1.0, step / 100) * (1.0 if step < max_steps * 0.8 else max(0.1, (max_steps - step) / (max_steps * 0.2)))
        for g in opt.param_groups:
            g["lr"] = lr * m
        x, y = data.batch(batch, T, rng)
        mix = pool(x.to(pool.dev), update_stats=False)
        loss = F.nll_loss(mix.reshape(-1, mix.size(-1)), y.to(pool.dev).reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        run.append(float(loss))
        if step % eval_every == 0 or step == 1:
            vt, wt = eval_pool(pool, val["ts"], pool.dev); vo, wo = eval_pool(pool, val["tool"], pool.dev)
            vm = 0.4 * vt + 0.6 * vo; hist.append((step, vm))
            log(f"[router] step {step} train {np.mean(run[-eval_every:]):.4f} | val ts {vt:.4f} tool {vo:.4f} | weights ts "
                + " ".join(f"{n}={w:.2f}" for n, w in zip(pool.names, wt)) + " | tool " + " ".join(f"{n}={w:.2f}" for n, w in zip(pool.names, wo))
                + f" | {time.time()-t0:.0f}s")
            with open(mp, "a", newline="") as fh:
                csv.writer(fh).writerow([step, round((time.time() - t0) / 60, 2), round(float(np.mean(run[-eval_every:])), 4), round(vt, 4), round(vo, 4)]
                                        + [round(w, 4) for w in wt] + [round(w, 4) for w in wo])
            old = [v for s_, v in hist if s_ <= step - window]
            if old and step < max_steps * 0.8 and (old[-1] - vm) / old[-1] < patience_rel:
                log(f"[router] flattened at step {step} (val gain over last {window} steps < {patience_rel*100:.1f}%)"); break
    for p in pool.router.parameters():
        p.requires_grad_(False)
    pool.eval()
    return dict(steps=step, seconds=round(time.time() - t0), val=hist[-1][1] if hist else None)


# ----------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description="Seed Pool v1")
    ap.add_argument("cmd", choices=["init", "train", "chat", "list", "add", "remove", "calibrate"])
    ap.add_argument("name", nargs="?"); ap.add_argument("path", nargs="?")
    ap.add_argument("--dir", default=str(POOL_DIR)); ap.add_argument("--device", default="cuda")
    ap.add_argument("--desc", default=""); ap.add_argument("--budget-mb", type=float, default=None)
    ap.add_argument("--max-steps", type=int, default=4000); ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--prompt", default=None); ap.add_argument("--tools", action="store_true"); ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--temp", type=float, default=0.7)
    a = ap.parse_args(); d = Path(a.dir); d.mkdir(parents=True, exist_ok=True)
    if a.cmd == "init":
        pool = SeedPool(a.budget_mb or 2048, a.device).to(a.device)
        pool.add_seed("run3", r"D:\seed_v4\run3\latest.pt", "fluent TinyStories trunk (Seed v4 run3, 4x2, organs o1+o2)")
        pool.add_seed("run4", r"D:\seed_v4\run4\latest.pt", "tools + reasoning seed (Seed v4 run4, 6x1): wiki_search/calculator calls, gsm8k-style steps")
        pool.save(d); print(json.dumps(pool.meta, indent=1)); return
    pool = load_pool(d, a.device, a.budget_mb)
    if a.cmd == "list":
        reg = json.loads((d / "registry.json").read_text())
        for n, s in reg["seeds"].items():
            print(f"{n:10s} {'ACTIVE' if n in pool.seeds else 'idle  '} {s['weights_mb']:7.1f} MB  usefulness {pool.stats.get(n, 0):.3f}  {s.get('description','')}")
        print(f"active weights {sum(pool.seed_mb(s) for s in pool.seeds.values()):.1f} MB of budget {pool.budget_mb:.0f} MB; router {pool.seed_mb(pool.router):.1f} MB")
    elif a.cmd == "train":
        logf = open(d / "train.log", "a", encoding="utf-8")
        def lg(s):
            print(s, flush=True); logf.write(f"[{time.strftime('%H:%M:%S')}] {s}\n"); logf.flush()
        r = train_router(pool, d, max_steps=a.max_steps, log=lg); pool.save(d); lg(json.dumps(r))
    elif a.cmd == "add":
        pool.add_seed(a.name, a.path, a.desc); pool.save(d)
    elif a.cmd == "remove":
        pool.remove_seed(a.name); pool.save(d)
    elif a.cmd == "calibrate":
        r = train_router(pool, d, max_steps=a.steps, only_keys=[a.name], eval_every=50, lr=1e-2); pool.save(d); print(r)
    elif a.cmd == "chat":
        tok = S.find_tokenizer(next(iter(pool.seeds.values())))
        def reply(text):
            ids = tok.encode(text).ids
            out = pool.generate(ids, a.n, temp=a.temp, stops=("<|end|>", "</tool_call>") if a.tools else (), tok=tok)
            return tok.decode(out)
        def turn(q):
            if not a.tools:
                return reply(q)
            sys.path.insert(0, r"D:\tiny100m\src"); import chat100m as C
            import re
            text = "<|system|>You are a helpful assistant. You can call these tools:\n" + "\n".join(json.dumps(v) for v in C.SPECS.values()) + "<|end|>\n"
            text += f"<|user|>{q}<|end|>\n<|assistant|>"; full = ""
            for _ in range(3):
                new = reply(text); text += new; full += new
                mt = re.search(r"<tool_call>(.*?)</tool_call>$", new, re.S)
                if not mt: break
                res = C.run_tool(mt.group(1), r"D:\tiny100m\workspace"); piece = f"<tool_result>{res}</tool_result>"; text += piece; full += piece
            return full
        if a.prompt:
            print(turn(a.prompt)); return
        while True:
            try: q = input("> ")
            except (EOFError, KeyboardInterrupt): break
            if q.strip() in ("/quit", "/q"): break
            print(turn(q))


if __name__ == "__main__":
    main()
