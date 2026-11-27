r"""Pool5 evaluation: held-out bpb across domains + optional Python coding eval.

    python pool5/eval/pool5_eval.py bpb  --dir pool5 --label top6 --tau 0.9 --out pool5/results
    python pool5/eval/pool5_eval.py code --dir pool5 --label top6 --tau 0.9 --out pool5/results

Domain files default to the original local training paths; pass --domains-json to point at your own held-out .txt files
(list of {name, files: [[path, k, whole], ...]}). The published results already live in pool5/results/.
"""
import sys, os, json, math, time, csv, argparse
from pathlib import Path
import torch, torch.nn.functional as F
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pool5_loop as L
RES = str(REPO / "pool5" / "results")
os.makedirs(RES, exist_ok=True)

def docs_from(path, k, whole=False, cap=4_000_000):
    if not os.path.exists(path): return []
    sz = os.path.getsize(path); f = open(path, "rb")
    if whole or sz <= cap: t = f.read(cap).decode("utf-8","ignore")
    else: f.seek(int(sz*0.995) - cap//2); t = f.read(cap).decode("utf-8","ignore")
    return [d for d in t.split("<|endoftext|>")[1:-1] if len(d) > 300][:k]

DOMAINS = [
 ("python", [(r"D:\seed_v4\seed_py1\data\public_py.txt",12,False),(r"D:\seed_v4\seed_py1\data\alpaca_py.txt",12,False)]),
 ("c",      [(r"D:\seed_v4\seed_c1\data\public_heldout.txt",24,True)]),
 ("js",     [(r"D:\seed_v4\seed_js1\data\public_heldout.txt",24,True)]),
 ("bash",   [(r"D:\seed_v4\seed_sh1\data\public_heldout.txt",24,True)]),
 ("powershell",[(r"D:\seed_v4\seed_ps1\data\public_heldout.txt",24,True)]),
 ("alg_real",[(r"D:\seed_v4\seed_alg1\data\real_heldout.txt",24,True)]),
 ("alg_synth",[(r"D:\seed_v4\seed_alg1\data\alg_heldout.txt",16,True)]),
 ("math",   [(r"D:\seed_v4\seed_math1\data\synth_math.txt",16,False)]),
 ("reason", [(r"D:\seed_v4\seed_reason1\data\synth_reason.txt",16,False)]),
 ("colby",  [(r"D:\seed_v4\seed_c1\data\colby_heldout.txt",8,True),(r"D:\seed_v4\seed_js1\data\colby_heldout.txt",8,True),
             (r"D:\seed_v4\seed_sh1\data\colby_heldout.txt",8,True),(r"D:\seed_v4\seed_ps1\data\colby_heldout.txt",8,True)]),
]

@torch.no_grad()
def adaptive_bpb(p5, X, tau):
    totnll = 0.0; totb = 0; psum = 0.0; ntok = 0
    for x, nbytes in X:
        outs, confs = L.all_passes(p5, x)           # outs[r]:(1,T,V); confs[r]:(1,T)
        y = x[0,1:]; Tm = y.numel()
        tgt = torch.stack([o[0,:-1].gather(-1, y[:,None]).squeeze(-1) for o in outs], 0)  # (P,Tm)
        chosen = torch.full((Tm,), L.MAXP-1, dtype=torch.long, device=x.device)
        for r in range(L.MAXP-2, -1, -1):
            take = confs[r][0,:-1] >= tau; chosen[take] = r
        lp = tgt.gather(0, chosen[None]).squeeze(0)
        totnll += float(-lp.sum()); totb += nbytes; psum += float((chosen+1).sum()); ntok += Tm
    return totnll/math.log(2)/max(1,totb), psum/max(1,ntok)

def run_bpb(dir, label, tau):
    p5 = L.load(dir, "cuda", with_loop=True); tok = __import__("seed_v4").find_tokenizer(next(iter(p5.seeds.values())))
    res = {"variant": label, "cfg": p5.cfg, "tau": tau, "domains": {}}
    for name, specs in DOMAINS:
        D = []
        for path,k,whole in specs: D += docs_from(path,k,whole)
        if not D: res["domains"][name] = None; continue
        X = []
        for d in D:
            ids = tok.encode(d).ids[:512]
            if len(ids) < 8: continue
            x = torch.tensor([ids], device="cuda"); X.append((x, len(tok.decode(ids).encode())))
        t=time.time(); bpb, mp = adaptive_bpb(p5, X, tau)
        res["domains"][name] = dict(bpb=round(bpb,4), mean_passes=round(mp,3), ndocs=len(X), sec=round(time.time()-t,1))
        print(f"{label} bpb {name:11} {bpb:.4f}  mean_passes {mp:.2f}  ({len(X)} docs, {time.time()-t:.0f}s)", flush=True)
    vals = [d["bpb"] for d in res["domains"].values() if d]; res["macro_bpb"] = round(sum(vals)/len(vals),4)
    json.dump(res, open(os.path.join(RES, f"{label}_bpb.json"),"w"), indent=1)
    print(f"{label} MACRO bpb {res['macro_bpb']}", flush=True); return res

def sample_logits(lg, hist, rp, temp, top_p):
    if rp and rp != 1.0:
        for t in set(hist): lg[t] = lg[t]/rp if lg[t] > 0 else lg[t]*rp
    lg = lg/max(temp,1e-5); p = F.softmax(lg,-1)
    sp, si = torch.sort(p, descending=True); cdf = sp.cumsum(-1); mask = cdf - sp > top_p
    sp[mask] = 0.0; sp = sp/sp.sum(); return int(si[torch.multinomial(sp,1)])

def run_code(dir, label, tau, sample):
    from problems import PROBLEMS, extract_code, run_tests
    try:
        from extract_v2 import extract_code_v2
    except Exception: extract_code_v2 = None
    import seed_v4 as S
    p5 = L.load(dir, "cuda", with_loop=True); tok = S.find_tokenizer(next(iter(p5.seeds.values()))); eos = tok.token_to_id("<eos>")
    stats = {"passes":[0]*(L.MAXP+1), "seed_runs":0, "routed":{}}
    @torch.no_grad()
    def gen(text, n=256, stops=()):
        ids = tok.encode(text).ids; out=[]
        for _ in range(n):
            x = torch.tensor([(ids+out)[-p5.ctx:]], device="cuda")
            lg = L.step_last(p5, x, tau, stats)
            nx = sample_logits(lg.clone(), out[-64:], 1.15, 0.8, 0.95) if sample else int(torch.argmax(lg))
            out.append(nx)
            if nx == eos: break
            t = tok.decode(out)
            if any(s in t for s in stops): break
        t = tok.decode(out)
        for s in stops:
            if s in t: t = t[:t.index(s)]
        return t
    rows=[]; t0=time.time(); ntok=0
    for name, p, tests in PROBLEMS:
        for mode in ("chat","complete"):
            if mode=="chat":
                pr = f"<|user|>Complete the following Python function. Reply with the complete function in one ```python code block.\n\n```python\n{p}```<|end|>\n<|assistant|>"
                out = gen(pr, stops=("<|end|>",))
            else:
                pr = p; out = gen(pr, stops=("\ndef ","\nclass ","\nif __name__","\nprint(","<|end|>","\n#"))
            ntok += len(tok.encode(out).ids)
            code = extract_code(out, p, name); ok, err = run_tests(code, tests)
            if not ok and extract_code_v2:
                c2 = extract_code_v2(out, p, name); ok2,_ = run_tests(c2, tests); ok = ok or ok2
            rows.append(dict(model=label, mode=mode, item=name, passed=ok))
    dt = time.time()-t0; npass = {m: sum(r["passed"] for r in rows if r["mode"]==m) for m in ("chat","complete")}
    tag = "_sample" if sample else ""
    with open(os.path.join(RES, f"{label}_code{tag}.csv"),"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    tot = sum(stats["passes"]) or 1
    summ = dict(variant=label, sampling=bool(sample), tau=tau, pass_chat=npass["chat"], pass_complete=npass["complete"],
                pass_best=max(npass.values()), n=20, sec=round(dt,1), tok_s=round(ntok/dt,1),
                mean_passes=round(sum(i*c for i,c in enumerate(stats["passes"]))/tot,3),
                pass_dist=stats["passes"], seed_runs=stats["seed_runs"],
                routing=dict(sorted(stats["routed"].items(), key=lambda kv:-kv[1])))
    json.dump(summ, open(os.path.join(RES, f"{label}_code{tag}.json"),"w"), indent=1)
    print(f"{label} code{tag}: chat {npass['chat']}/20 complete {npass['complete']}/20 | {summ['tok_s']} tok/s mean_passes {summ['mean_passes']}", flush=True)
    print(f"  routing {summ['routing']}", flush=True); return summ

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["bpb","code"]); ap.add_argument("--dir", required=True)
    ap.add_argument("--label", required=True); ap.add_argument("--tau", type=float, default=0.9); ap.add_argument("--sample", action="store_true")
    a = ap.parse_args()
    if a.cmd=="bpb": run_bpb(a.dir, a.label, a.tau)
    else: run_code(a.dir, a.label, a.tau, a.sample)
if __name__=="__main__": main()
