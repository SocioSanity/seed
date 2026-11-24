"""pool2 test (made from pool1/test_pool1.py by make_test2.py; same harness): run4, pool1, pool2. Original header follows. merged1 test: run3 alone, run4 alone, merged1 (run3 + run4 organ), tiny100m (ckpt.pt step 1361, EMA).
1) bits per byte on the same held-out text (raw model, no decode prior)  2) functional suite with real tools (chat100m harness)
3) params / weights MB / RAM / inference tok/s.  One run; writes D:\seed_v4\merged1\results\*."""
import os, sys, json, time, re, math, csv, random, gc
import numpy as np, torch, torch.nn.functional as F, psutil
torch.cuda.init()
sys.path.insert(0, r"C:\Users\mcgar\Desktop\Seed"); sys.path.insert(0, r"D:\tiny100m\src")
import seed_v4 as S
from tokenizers import Tokenizer
import chat100m as C          # sets CUDA_VISIBLE_DEVICES="" but CUDA is already initialised
from model100m import Tiny100M
R = r"D:\seed_v4\pool2\results"; dev = "cuda"; LN2 = math.log(2); t_start = time.time()
proc = psutil.Process()

# ------------------------------------------------------------------ models
def load_seed(path):
    m, info = S.load_model_any(path, use_ema=True, device=dev)
    tok = S.find_tokenizer(m)
    return dict(kind="seed", m=m, tok=tok, W=m.ctx, eos=tok.token_to_id("<eos>"), step=info["step"])

def load_100m(path):
    s = torch.load(path, map_location="cpu", weights_only=False)
    m = Tiny100M(s["core_cfg"], s["vocab"], block=s["block"]); m.load_state_dict(s.get("ema") or s["model"]); m.eval().to(dev)
    tok = Tokenizer.from_file(r"D:\tiny100m\tokenizer\tokenizer.json")
    st = s.get("step"); del s; gc.collect()
    return dict(kind="100m", m=m, tok=tok, W=s_block if False else m.block, eos=3, step=st)

import seed_pool as P
def load_poolN(d, name):
    pool = P.load_pool(d, dev); pool.eval()
    tok = S.find_tokenizer(next(iter(pool.seeds.values())))
    return dict(kind="seed", m=pool, tok=tok, W=pool.ctx, eos=tok.token_to_id("<eos>"), step="router", pool=True, name=name)
# same harness as the merged1 test; the 100M's numbers are reused from D:\seed_v4\merged1\results (harness unchanged)
MODELS = [("run4", lambda: load_seed(r"D:\seed_v4\merged1\work\run4_src.pt")),
          ("pool1", lambda: load_poolN(r"D:\seed_v4\pool1", "pool1")),
          ("pool2", lambda: load_poolN(r"D:\seed_v4\pool2", "pool2"))]
REGION = {}   # pool1 router weight on run4, by text region
def router_regions(M, prompt, reply, kind):
    if not M.get("pool"): return
    text = prompt + reply; enc = M["tok"].encode(text); ids = enc.ids
    with torch.no_grad():
        _, w = M["m"](torch.tensor([ids], device=dev), return_weights=True, update_stats=True)
    names = list(M["m"].names); W_ = w[0].float().cpu().numpy()
    spans = []
    for tag, rg in (("think", r"<think>.*?</think>"), ("tool_call", r"<tool_call>.*?</tool_call>"), ("tool_result", r"<tool_result>.*?</tool_result>")):
        for mm in re.finditer(rg, reply, re.S): spans.append((len(prompt) + mm.start(), len(prompt) + mm.end(), tag))
    for k, (a, b) in enumerate(enc.offsets):
        if a < len(prompt): continue
        reg = "story" if kind.startswith("story") else next((t for s0, s1, t in spans if s0 <= a < s1), "answer")
        # the weight at position k-1 predicts token k
        if k >= 1:
            for jn, nm in enumerate(names): REGION.setdefault((M["name"], kind.split("_")[0], reg, nm), []).append(float(W_[k - 1, jn]))

def logits_of(M, x):
    out = M["m"](x)
    return (out[0] if isinstance(out, tuple) else out).float()

# ------------------------------------------------------------------ bpb data
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

SRC = ["wiki_tool", "glaive", "instruct", "calc", "tools_misc", "gsm8k"]
SETS = {}
for s_ in SRC:
    SETS[("seed_heldout", s_)] = tail_docs(s_, 60)
    SETS[("100m_val", s_)] = val100_docs(s_, 60)
ts = open(r"D:\datasets\TinyStoriesV2-GPT4-valid.txt", encoding="utf-8").read(3_000_000).split("<|endoftext|>")
SETS[("ts_valid", "TinyStories")] = [d.strip() for d in ts[:200] if len(d.strip()) > 50]

@torch.no_grad()
def doc_bits(M, text):
    ids = [M["eos"]] + M["tok"].encode(text).ids
    N = len(ids); W = M["W"]; St = W // 2; tot = 0.0
    a = 1
    while a < N:
        b = min(N, (W if a == 1 else a + St))
        lo = max(0, b - W)
        x = torch.tensor([ids[lo:b]], device=dev)
        lg = logits_of(M, x)[0]
        tgt = torch.tensor(ids[a:b], device=dev)
        lp = F.log_softmax(lg[a - lo - 1: b - lo - 1], -1)
        tot += float(-lp.gather(1, tgt[:, None]).sum())
        a = b
    return tot / LN2, len(text.encode("utf-8")), N - 1

# ------------------------------------------------------------------ functional suite
SYSTEM = "<|system|>You are a helpful assistant. You can call these tools:\n" + "\n".join(json.dumps(v) for v in C.SPECS.values()) + "<|end|>\n"
FACT = [("Who was Albert Einstein?", ["einstein"], ["physic"]), ("Who was Marie Curie?", ["curie"], ["physic", "chemist", "radioactiv", "nobel"]),
        ("What is the Eiffel Tower?", ["eiffel"], ["paris", "tower"]), ("What is the Amazon River?", ["amazon"], ["south america", "river"]),
        ("Who was Isaac Newton?", ["newton"], ["physic", "mathematic", "gravit"]), ("What is Mount Everest?", ["everest"], ["mountain", "himalaya", "highest"]),
        ("Who was William Shakespeare?", ["shakespeare"], ["playwright", "poet", "writer"]), ("What is Python (programming language)?", ["python"], ["programming"]),
        ("Who was Leonardo da Vinci?", ["vinci", "leonardo"], ["painter", "artist", "italian", "polymath"]), ("What is the Great Wall of China?", ["great wall"], ["china", "fortification", "wall"]),
        ("Who was Abraham Lincoln?", ["lincoln"], ["president"]), ("What is photosynthesis?", ["photosynthes"], ["light", "plant", "energy"]),
        ("What is Mars?", ["mars"], ["planet"]), ("Who was Nelson Mandela?", ["mandela"], ["south africa", "president"]),
        ("What is Tokyo?", ["tokyo"], ["japan", "capital"]), ("Who was Charles Darwin?", ["darwin"], ["evolution", "natural", "naturalist", "biologist"]),
        ("Who were the Beatles?", ["beatles"], ["band", "liverpool", "rock"]), ("What is the Moon?", ["moon"], ["satellite", "earth"]),
        ("Who was Pablo Picasso?", ["picasso"], ["painter", "artist", "spanish"]), ("Who was Wolfgang Amadeus Mozart?", ["mozart"], ["composer"])]
MATH = [("Compute 642 + 135", "777"), ("What is 347 * 29?", "10063"), ("Compute 1250 - 487", "763"), ("What is 144 / 12?", "12"), ("Compute 23 * 17 + 9", "400"),
        ("Tom has 12 apples. He gives 5 to his sister and buys 8 more. How many apples does Tom have now?", "15"),
        ("A shop sells pencils for 3 dollars each. Sara buys 7 pencils and pays with a 50 dollar bill. How much change does she get?", "29"),
        ("There are 4 boxes with 18 books in each box. How many books are there in total?", "72"),
        ("Mike runs 5 miles every day for 2 weeks. How many miles does he run in total?", "70"),
        ("A baker makes 120 cookies and sells 3/4 of them. How many cookies are left?", "30")]
OBSCURE = json.load(open(r"D:\seed_v4\pool2\heldout_obscure_facts.json", encoding="utf-8"))
STORY = [("plain", "Once upon a time, a little girl named Mia found a small blue egg in the garden."),
         ("chat", "<|user|>Tell me a short story about a dog who is afraid of the dark.<|end|>\n<|assistant|>")]

@torch.no_grad()
def sample(M, text, max_new=200, temp=0.0, top_k=40, stops=("<|end|>", "</tool_call>"), seed=0):
    torch.manual_seed(seed)
    tok = M["tok"]; ids = tok.encode(text).ids; out = []
    pr = M["m"].__dict__.get("decode_prior") if M["kind"] == "seed" else None
    for _ in range(max_new):
        x = torch.tensor([(ids + out)[-M["W"]:]], device=dev)
        lg = logits_of(M, x)[0, -1]
        if pr is not None:
            lg = lg + S.prior_bias(pr, (ids + out)[-1], lg.numel(), lg.device)
        if temp <= 0: nxt = int(torch.argmax(lg))
        else:
            lg = lg / temp; v, _ = torch.topk(lg, top_k); lg[lg < v[-1]] = -float("inf")
            nxt = int(torch.multinomial(F.softmax(lg, -1), 1))
        out.append(nxt)
        if M["kind"] == "100m" and nxt in (tok.token_to_id("<|end|>"), tok.token_to_id("</tool_call>")):
            break
        tail = tok.decode(out[-8:])
        if M["kind"] == "seed" and (nxt == M["eos"] or any(tail.endswith(s_) for s_ in stops)):
            break
    return tok.decode(out, skip_special_tokens=False) if M["kind"] == "100m" else tok.decode(out)

def respond(M, q, seed):
    text = SYSTEM + f"<|user|>{q}<|end|>\n<|assistant|>"; reply = ""; calls = []
    for _ in range(3):
        new = sample(M, text, seed=seed); text += new; reply += new
        mt = re.search(r"<tool_call>(.*?)</tool_call>$", new, re.S)
        if mt:
            body = mt.group(1); ok = True
            try:
                c = json.loads(body); ok = isinstance(c, dict) and "name" in c and "arguments" in c
            except Exception:
                ok = False; c = {}
            res = C.run_tool(body, r"D:\tiny100m\workspace") if ok else "Error: could not parse tool call"
            args = c.get("arguments", {}) if isinstance(c, dict) else {}
            if isinstance(args, str):
                try: args = json.loads(args)
                except Exception: args = {}
            calls.append(dict(raw=body, valid=ok, name=c.get("name") if isinstance(c, dict) else None, args=args, result=res[:300]))
            piece = f"<tool_result>{res}</tool_result>"; text += piece; reply += piece
            continue
        break
    final = re.sub(r"<think>.*?</think>|<tool_call>.*?</tool_call>|<tool_result>.*?</tool_result>", "", reply, flags=re.S).replace("<|end|>", "").strip()
    return reply, final, calls

# ------------------------------------------------------------------ run
bpb_rows, fun_rows, trans, specs = [], [], [], []
for name, loader in MODELS:
    rss0 = proc.memory_info().rss; t0 = time.time()
    M = loader(); torch.cuda.synchronize()
    npar = sum(p.numel() for p in M["m"].parameters())
    print(f"== {name} step {M['step']} params {npar:,} window {M['W']}", flush=True)
    # bpb
    for (setname, src), docs in SETS.items():
        bits = bys = toks = 0
        for d in docs:
            b_, y_, t_ = doc_bits(M, d); bits += b_; bys += y_; toks += t_
        bpb_rows.append(dict(model=name, set=setname, source=src, docs=len(docs), bytes=bys, tokens=toks, bpb=round(bits / bys, 4)))
    print("  bpb done", round(time.time() - t0), "s", flush=True)
    # functional
    for i, (q, qkeys, akeys) in enumerate(FACT):
        reply, final, calls = respond(M, q, seed=i)
        router_regions(M, SYSTEM + f"<|user|>{q}<|end|>\n<|assistant|>", reply, "fact")
        wiki = [c for c in calls if c["name"] == "wiki_search"]
        qtxt = " ".join(str(c["args"].get("query", "")) for c in wiki).lower()
        fun_rows.append(dict(model=name, kind="fact", item=q, n_calls=len(calls), valid_calls=sum(c["valid"] for c in calls), used_wiki=bool(wiki),
                             query_ok=any(k in qtxt for k in qkeys), answer_ok=any(k in final.lower() for k in akeys), final=final[:300]))
        trans.append(dict(model=name, kind="fact", q=q, reply=reply))
    for i, ob in enumerate(OBSCURE):
        q, qkeys, akeys = ob["q"], ob["qkeys"], ob["akeys"]
        reply, final, calls = respond(M, q, seed=300 + i)
        router_regions(M, SYSTEM + f"<|user|>{q}<|end|>\n<|assistant|>", reply, "obscure")
        wiki = [c for c in calls if c["name"] == "wiki_search"]
        qtxt = " ".join(str(c["args"].get("query", "")) for c in wiki).lower()
        fun_rows.append(dict(model=name, kind="obscure", item=q, n_calls=len(calls), valid_calls=sum(c["valid"] for c in calls), used_wiki=bool(wiki),
                             query_ok=any(k in qtxt for k in qkeys), answer_ok=any(k in final.lower() for k in akeys), final=final[:300]))
        trans.append(dict(model=name, kind="obscure", q=q, reply=reply))
    for i, (q, ans) in enumerate(MATH):
        reply, final, calls = respond(M, q, seed=100 + i)
        router_regions(M, SYSTEM + f"<|user|>{q}<|end|>\n<|assistant|>", reply, "math")
        nums = re.findall(r"-?\d+(?:\.\d+)?", final.replace(",", ""))
        fun_rows.append(dict(model=name, kind="math", item=q, n_calls=len(calls), valid_calls=sum(c["valid"] for c in calls), used_wiki=False,
                             query_ok="", answer_ok=ans in nums, final=final[:300]))
        trans.append(dict(model=name, kind="math", q=q, reply=reply))
    for i, (kind, p) in enumerate(STORY):
        out = sample(M, p, max_new=120, seed=200 + i, stops=("<|end|>",))
        router_regions(M, p, out, "story_" + kind)
        trans.append(dict(model=name, kind="story_" + kind, q=p, reply=out))
    # speed + memory (GPU and CPU, batch 1, 128 new tokens, no stops)
    def speed(d):
        M["m"].to(d); x = M["tok"].encode("Once upon a time").ids
        torch.manual_seed(0); t1 = time.time()
        with torch.no_grad():
            ids = x[:]
            for _ in range(128):
                lg = logits_of(M, torch.tensor([ids[-M["W"]:]], device=d))[0, -1]
                ids.append(int(torch.argmax(lg)))
        if d == "cuda": torch.cuda.synchronize()
        return 128 / (time.time() - t1)
    torch.set_num_threads(4)
    g_tps = speed("cuda"); c_tps = speed("cpu")
    rss = (proc.memory_info().rss - rss0) / 2**20
    specs.append(dict(model=name, step=M["step"], params=npar, weights_mb_fp32=round(npar * 4 / 2**20, 1), weights_mb_fp16=round(npar * 2 / 2**20, 1),
                      ram_delta_mb_cpu=round(rss), context=M["W"], tok_s_gpu=round(g_tps, 1), tok_s_cpu4=round(c_tps, 1)))
    print("  ", specs[-1], f"total {round(time.time() - t0)} s", flush=True)
    del M; gc.collect(); torch.cuda.empty_cache()

def wcsv(fn, rows):
    with open(os.path.join(R, fn), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
wcsv("bpb.csv", bpb_rows); wcsv("functional.csv", fun_rows); wcsv("models.csv", specs)
with open(os.path.join(R, "transcripts.jsonl"), "w", encoding="utf-8") as f:
    for t in trans: f.write(json.dumps(t, ensure_ascii=False) + "\n")
summ = []
for name, _ in MODELS:
    fr = [r for r in fun_rows if r["model"] == name]; fa = [r for r in fr if r["kind"] == "fact"]; ma = [r for r in fr if r["kind"] == "math"]; ob = [r for r in fr if r["kind"] == "obscure"]
    nc = sum(r["n_calls"] for r in fr); vc = sum(r["valid_calls"] for r in fr)
    row = dict(model=name, tool_calls=nc, valid_json_rate=round(vc / nc, 3) if nc else 0, fact_used_wiki=sum(r["used_wiki"] for r in fa),
               fact_query_ok=sum(bool(r["query_ok"]) for r in fa), fact_answer_ok=sum(r["answer_ok"] for r in fa), math_ok=sum(r["answer_ok"] for r in ma),
               math_used_calc=sum(r["n_calls"] > 0 for r in ma), obscure_used_wiki=sum(r["used_wiki"] for r in ob),
               obscure_query_ok=sum(bool(r["query_ok"]) for r in ob), obscure_answer_ok=sum(r["answer_ok"] for r in ob), decoding="greedy")
    for (setname, src) in SETS:
        b = [r for r in bpb_rows if r["model"] == name and r["set"] == setname and r["source"] == src][0]
        row[f"bpb_{setname}_{src}"] = b["bpb"]
    summ.append(row)
wcsv("summary.csv", summ)
rr = [dict(model=k[0], task=k[1], region=k[2], seed=k[3], tokens=len(v), mean_w=round(float(np.mean(v)), 4), frac_gt_0p5=round(float(np.mean(np.array(v) > 0.5)), 3)) for k, v in sorted(REGION.items())]
if rr: wcsv("router_regions.csv", rr); print(json.dumps(rr, indent=1))
print(json.dumps(summ, indent=1)); print("total seconds", round(time.time() - t_start))
