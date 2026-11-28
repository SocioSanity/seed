#!/usr/bin/env python3
"""Exact-match eval for reason1-cael-undertrained.

Greedy, right-padded batched decode (no KV cache). Reports exact match of the
"Answer:" line on official CAEL test/hard (stratified by category) and human-written
GSM8K test. If a CAEL verify.py is on --verify-dir, also reports verify.judge.

Copies the checkpoint before loading so a live trainer can keep writing it.

    python recipes/reason1_cael/eval_r1.py path/to/best_ce.pt tag --n-cat 30 --n-hard 15 --n-gsm 100
"""
from __future__ import annotations
import argparse, collections, json, os, random, re, shutil, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "v2"))
import torch
import seed_v2 as sv
from tokenizers import Tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("tag")
ap.add_argument("--n-cat", type=int, default=30)
ap.add_argument("--n-hard", type=int, default=15)
ap.add_argument("--n-gsm", type=int, default=100)
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--max-new", type=int, default=200)
ap.add_argument("--tokenizer", default=str(REPO / "tokenizers" / "seed-bpe8192-r1" / "tokenizer.json"))
ap.add_argument("--cael-text", default=r"D:\seed_v2\data\cael\text")
ap.add_argument("--gsm8k-eval", default=r"D:\seed_v2\data\gsm8k\eval_gsm8k_test.jsonl")
ap.add_argument("--verify-dir", default=r"D:\seed_v2\data\cael\raw", help="directory containing CAEL verify.py (optional)")
ap.add_argument("--out-dir", default=None)
a = ap.parse_args()
torch.set_num_threads(a.threads)
sv.DEV = torch.device("cpu")
tok = Tokenizer.from_file(a.tokenizer)
EOS = tok.token_to_id("<eos>")

judge = None
if a.verify_dir and (Path(a.verify_dir) / "verify.py").exists():
    sys.path.insert(0, a.verify_dir)
    import verify as _verify
    judge = _verify.judge

out_dir = Path(a.out_dir or (Path(a.ckpt).resolve().parent / "eval"))
out_dir.mkdir(parents=True, exist_ok=True)
tmp = out_dir / f"_ckpt_{a.tag}.pt"
shutil.copyfile(a.ckpt, tmp)
c = torch.load(tmp, map_location="cpu", weights_only=False)
model = sv.SeedV2.from_structure(c["structure"])
model.load_state_dict(c["model"])
if c.get("ema"):
    e = sv.EMA(model, 0.999)
    e.load_by_name(model, c["ema"])
    e.swap(model)
step = c.get("S", {}).get("step", c.get("step"))
print("loaded", a.ckpt, "step", step, "params", sum(p.numel() for p in model.parameters()), flush=True)
os.remove(tmp)


@torch.no_grad()
def decode(prompts):
    outs = [None] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    for s in range(0, len(order), a.bs):
        idxs = order[s:s + a.bs]
        seqs = [tok.encode(prompts[i], add_special_tokens=False).ids[-(model.ctx - 8):] for i in idxs]
        gen = [[] for _ in idxs]
        done = [False] * len(idxs)
        for _ in range(a.max_new):
            L = max(len(q) + len(g) for q, g in zip(seqs, gen))
            if L >= model.ctx:
                break
            x = torch.zeros(len(idxs), L, dtype=torch.long)
            for r, (q, g) in enumerate(zip(seqs, gen)):
                t = q + g
                x[r, :len(t)] = torch.tensor(t)
            z = model.trunk(x, None, None)
            for r, (q, g) in enumerate(zip(seqs, gen)):
                if done[r]:
                    continue
                nxt = int(sv._ce_logits(z[r, len(q) + len(g) - 1][None], model.E.weight, model.softcap)[0].argmax())
                if nxt == EOS:
                    done[r] = True
                    continue
                g.append(nxt)
                txt = tok.decode(g)
                if re.search(r"\nAnswer:[^\n]*\n", txt):
                    done[r] = True
            if all(done):
                break
        for r, i in enumerate(idxs):
            outs[i] = tok.decode(gen[r])
    return outs


def norm(s):
    return re.sub(r"\s+", " ", s.strip().rstrip(".").lower())


def final_of(txt):
    m = re.findall(r"(?im)^answer:\s*(.*)$", txt)
    return m[-1].strip() if m else ""


res = dict(ckpt=a.ckpt, step=step, structure=c.get("structure"), tag=a.tag, time=time.strftime("%Y-%m-%d %H:%M:%S"))
rnd = random.Random(2026)


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf8")]

for split, n in [("test", a.n_cat), ("hard", a.n_hard)]:
    rows = [r for r in load(str(Path(a.cael_text) / f"eval_{split}.jsonl")) if r.get("toks", 0) <= 480]
    bycat = collections.defaultdict(list)
    for r in rows:
        bycat[r["category"]].append(r)
    sel = []
    for cat in sorted(bycat):
        rr = bycat[cat][:]
        rnd.shuffle(rr)
        sel += rr[:n]
    t0 = time.time()
    outs = decode([r["prompt"] for r in sel])
    per = collections.defaultdict(lambda: [0, 0, 0])
    samples = []
    for r, o in zip(sel, outs):
        st_hit = norm(final_of(o)) == norm(r["answer"])
        j_hit = st_hit
        if judge is not None:
            j_hit = judge(o, dict(category=r["category"], answer=r["answer"]))["status"] == "correct"
        p = per[r["category"]]
        p[0] += 1
        p[1] += int(j_hit)
        p[2] += int(st_hit)
        if len(samples) < 6:
            samples.append(dict(cat=r["category"], prompt=r["prompt"][-200:], gold=r["answer"], out=o[:300]))
    tot = [sum(v[i] for v in per.values()) for i in range(3)]
    res[split] = dict(
        n=tot[0],
        judge_acc=round(tot[1] / max(1, tot[0]), 4),
        strict_acc=round(tot[2] / max(1, tot[0]), 4),
        per_category={k: dict(n=v[0], judge=round(v[1] / v[0], 3), strict=round(v[2] / v[0], 3)) for k, v in per.items()},
        secs=round(time.time() - t0),
        samples=samples,
    )
    print(split, json.dumps({k: v for k, v in res[split].items() if k != "samples"}), flush=True)

g = load(a.gsm8k_eval)
rnd.shuffle(g)
g = g[: a.n_gsm]
t0 = time.time()
outs = decode([r["prompt"] for r in g])
ok, ex = 0, []
for r, o in zip(g, outs):
    f = final_of(o).replace(",", "").replace("$", "").strip().rstrip(".")
    hit = f == r["answer"]
    ok += int(hit)
    if len(ex) < 4:
        ex.append(dict(gold=r["answer"], out=o[:300]))
res["gsm8k_test"] = dict(n=len(g), exact_match=round(ok / max(1, len(g)), 4), secs=round(time.time() - t0), samples=ex)
print("gsm8k_test", res["gsm8k_test"]["n"], res["gsm8k_test"]["exact_match"], flush=True)
json.dump(res, open(out_dir / f"eval_{a.tag}.json", "w"), indent=1)
print("wrote", out_dir / f"eval_{a.tag}.json")
