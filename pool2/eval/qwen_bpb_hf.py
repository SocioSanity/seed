"""Qwen2.5-Coder-3B-Instruct bits-per-byte on the exact pool2 held-out sets (same docs, same bytes as test_pool2.py doc selection).
True log-probs from HF weights (official Qwen/Qwen2.5-Coder-3B-Instruct, downloaded to D:). Run in D:\seed_v4\qwen_env (isolated venv).
Scoring = test harness rule: ids = [<|endoftext|>] + tokens(doc); every doc token scored (all docs fit in one window); bits / UTF-8 bytes."""
import os, sys, json, math, time, csv, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
DTYPE = os.environ.get("QDTYPE", "float16")
M = r"D:\seed_v4\qwen_env\models\Qwen2.5-Coder-3B-Instruct"; R = r"D:\seed_v4\pool2\results"; LN2 = math.log(2)
# --- identical doc selection to test_pool2.py
def tail_docs(name, n, maxb=2500):
    p = rf"D:\seed_v4\run4_data\{name}.txt"; size = os.path.getsize(p); cut = int(size * 0.99)
    with open(p, "rb") as f:
        f.seek(cut); raw = f.read().decode("utf-8", errors="ignore")
    ds = [d for d in raw.split("<|endoftext|>")[1:] if len(d) > 50][:n]
    return [d.encode()[:maxb].decode("utf-8", errors="ignore") for d in ds]
def val100_docs(name, n, maxb=2500):
    lines = open(rf"D:\tiny100m\corpus\{name}.jsonl", encoding="utf-8").read().splitlines()[-300:]
    ds = [json.loads(l)["text"] for l in lines][:n]
    return [d.encode()[:maxb].decode("utf-8", errors="ignore") for d in ds]
SETS = {}
for s_ in ["wiki_tool", "glaive", "instruct", "calc", "tools_misc", "gsm8k"]:
    SETS[("seed_heldout", s_)] = tail_docs(s_, 60); SETS[("100m_val", s_)] = val100_docs(s_, 60)
ts = open(r"D:\datasets\TinyStoriesV2-GPT4-valid.txt", encoding="utf-8").read(3_000_000).split("<|endoftext|>")
SETS[("ts_valid", "TinyStories")] = [d.strip() for d in ts[:200] if len(d.strip()) > 50]
ONLY = os.environ.get("QSETS")
if ONLY: SETS = {k: v for k, v in SETS.items() if k[1] in ONLY.split(",")}
OUT = os.environ.get("QOUT", "qwen_bpb.csv")
tok = AutoTokenizer.from_pretrained(M)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(M, dtype=getattr(torch, DTYPE), device_map="auto", max_memory={0: "5600MiB", "cpu": "11GiB"}, low_cpu_mem_usage=True).eval()
print("loaded", DTYPE, round(time.time() - t0), "s; device map:", dict(list({k: str(v) for k, v in model.hf_device_map.items()}.items())[-4:]), flush=True)
eos = tok.convert_tokens_to_ids("<|endoftext|>"); dev0 = next(model.parameters()).device
rows = []; nonfinite = 0
for (setname, src), docs in SETS.items():
    bits = bys = toks = 0; t1 = time.time()
    for d in docs:
        ids = [eos] + tok(d, add_special_tokens=False)["input_ids"]
        x = torch.tensor([ids], device="cuda")
        with torch.no_grad():
            lg = model(x).logits[0, :-1].float()
            lp = torch.log_softmax(lg, -1).gather(1, x[0, 1:, None].to(lg.device)).squeeze(1)
        if not torch.isfinite(lp).all(): nonfinite += 1
        bits += float(-lp.sum()) / LN2; bys += len(d.encode("utf-8")); toks += len(ids) - 1
    rows.append(dict(model="qwen2.5-coder-3b-instruct (HF " + DTYPE + ")", set=setname, source=src, docs=len(docs), bytes=bys, tokens=toks, bpb=round(bits / bys, 4)))
    print(rows[-1], f"{time.time()-t1:.0f}s", flush=True)
with open(os.path.join(R, OUT), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("nonfinite docs", nonfinite, "total", round(time.time() - t0), "s", flush=True)
