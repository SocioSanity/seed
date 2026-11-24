"""Fair Qwen baseline (Colby 2:01 PM): qwen2.5-coder:3b Q4_K_M (local Ollama GGUF blob) on CPU llama.cpp (llama-server -ngl 0 -sp), GREEDY (temp 0).
modes:  fewshot = our <think>/<tool_call>/<tool_result> format, tool schema + 3 few-shot examples in the system prompt
        native  = Qwen chat template: calls rendered <tool_call>..</tool_call>, results as tool turns (<|im_start|>user <tool_response>..), max 2 calls,
                  duplicate-call guard, then an explicit "now answer" step
        closed  = closed-book reference (no tools), plus the two story prompts
Same 20 facts, 15 held-out obscure facts, 10 math, real wiki_search/calculator. Writes D:\seed_v4\pool2\results\qwen_fair_<mode>_*."""
import os, sys, json, time, re, csv, subprocess, urllib.request
sys.path.insert(0, r"D:\tiny100m\src"); os.environ["CUDA_VISIBLE_DEVICES"] = ""
import chat100m as C
R = r"D:\seed_v4\pool2\results"; QN = "qwen2.5-coder:3b Q4_K_M"
GGUF = r"C:\Users\mcgar\.ollama\models\blobs\sha256-4a188102020e9c9530b687fd6400f775c45e90a0d7baafe65bd0a36963fbb7ba"
LS = r"C:\Users\mcgar\AppData\Local\Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-server.exe"; PORT = 8093
src = open(r"D:\seed_v4\pool2\test_pool2.py", encoding="utf-8").read()
ns = {"json": json}
exec(src[src.index("FACT = ["):src.index("@torch.no_grad()\ndef sample")], ns)
FACT, MATH, STORY, OBSCURE = ns["FACT"], ns["MATH"], ns["STORY"], ns["OBSCURE"]
SPECS = "\n".join(json.dumps(v) for v in C.SPECS.values())
FEWSHOT_SYS = ("You are a helpful assistant. You can call these tools:\n" + SPECS + "\n\n"
"Format rules: think briefly inside <think>...</think>. To call a tool write exactly <tool_call>{\"name\": \"<tool>\", \"arguments\": {...}}</tool_call> and stop; "
"the tool result will be inserted as <tool_result>...</tool_result>. After the result, write a short final answer in plain text.\n\n"
"Example 1\nUser: Who was Ada Lovelace?\nAssistant: <think>This is a factual question, I should search Wikipedia.</think><tool_call>{\"name\": \"wiki_search\", \"arguments\": {\"query\": \"Ada Lovelace\"}}</tool_call>"
"<tool_result>1. Ada Lovelace: Augusta Ada King, Countess of Lovelace (10 December 1815 – 27 November 1852) was an English mathematician and writer, chiefly known for her work on Charles Babbage's proposed mechanical general-purpose computer, the Analytical Engine.</tool_result>"
"Ada Lovelace was an English mathematician and writer known for her work on Charles Babbage's Analytical Engine.\n\n"
"Example 2\nUser: What is 48 * 25 + 7?\nAssistant: <think>I will use the calculator to be exact.</think><tool_call>{\"name\": \"calculator\", \"arguments\": {\"expression\": \"48 * 25 + 7\"}}</tool_call>"
"<tool_result>result: 1207</tool_result>The answer is 1207.\n\n"
"Example 3\nUser: What is the Danube?\nAssistant: <think>Let me look this up on Wikipedia.</think><tool_call>{\"name\": \"wiki_search\", \"arguments\": {\"query\": \"Danube\"}}</tool_call>"
"<tool_result>1. Danube: The Danube is the second-longest river in Europe, after the Volga in Russia. It flows through Central and Southeastern Europe, from the Black Forest south into the Black Sea.</tool_result>"
"The Danube is the second-longest river in Europe; it flows from the Black Forest to the Black Sea.")
NATIVE_SYS = ("You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
"You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n" + "\n".join(json.dumps({"type": "function", "function": v}) for v in C.SPECS.values())
+ "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>")
NOW_ANSWER = "Now answer my original question in one or two sentences, using the tool results above. Do not call any more tools."
CLOSED_SYS = "You are a helpful assistant."
def im(role, txt): return f"<|im_start|>{role}\n{txt}<|im_end|>\n"
def comp(prompt, n=200, stop=("<|im_end|>", "<|endoftext|>")):
    body = dict(prompt=prompt, n_predict=n, temperature=0.0, top_k=1, seed=0, stop=list(stop), cache_prompt=True)
    rq = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(rq, timeout=900).read()); return r["content"], r
def find_call(new):
    m = re.search(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", new, re.S)
    cands = [m.group(1)] if m else []
    dec = json.JSONDecoder()
    for mm in re.finditer(r"\{", new):
        try: obj, end = dec.raw_decode(new[mm.start():])
        except Exception: continue
        cands.append(new[mm.start():mm.start() + end]); break
    for c in cands:
        try:
            o = json.loads(c)
            if isinstance(o, dict) and o.get("name") in C.SPECS and "arguments" in o: return c, bool(m), new.find(c)
        except Exception: pass
    return (cands[0], bool(m), new.find(cands[0])) if m else (None, False, -1)
def do_call(body):
    try:
        c = json.loads(body); ok = isinstance(c, dict) and "name" in c and "arguments" in c
    except Exception: c, ok = {}, False
    res = C.run_tool(body, r"D:\tiny100m\workspace") if ok else "Error: could not parse tool call"
    args = c.get("arguments", {}) if isinstance(c, dict) else {}
    if isinstance(args, str):
        try: args = json.loads(args)
        except Exception: args = {}
    return dict(raw=body, valid=ok, name=c.get("name") if isinstance(c, dict) else None, args=args, result=res[:300]), res
def respond(mode, q):
    calls = []; reply = ""
    if mode == "closed":
        out, _ = comp(im("system", CLOSED_SYS) + im("user", q) + "<|im_start|>assistant\n"); return out, out.strip(), calls, dict(now_answer=False)
    if mode == "fewshot":
        text = im("system", FEWSHOT_SYS) + im("user", q) + "<|im_start|>assistant\n"
        for _ in range(3):
            new, r = comp(text, stop=("</tool_call>", "<|im_end|>", "<|endoftext|>"))
            if r.get("stopping_word") == "</tool_call>": new += "</tool_call>"
            text += new; reply += new
            m = re.search(r"<tool_call>(.*?)</tool_call>$", new, re.S)
            if not m: break
            c, res = do_call(m.group(1).strip()); c["tagged"] = True; calls.append(c)
            piece = f"<tool_result>{res}</tool_result>"; text += piece; reply += piece
        final = re.sub(r"<think>.*?</think>|<tool_call>.*?</tool_call>|<tool_result>.*?</tool_result>", "", reply, flags=re.S).strip()
        return reply, final, calls, dict(now_answer=False)
    # native
    text = im("system", NATIVE_SYS) + im("user", q) + "<|im_start|>assistant\n"; seen = set(); forced = False; final = ""
    for _ in range(2):
        new, r = comp(text)
        body, tagged, pos = find_call(new)
        if body is None:
            reply += new; final = new.strip(); break
        pre = new[:pos].replace("<tool_call>", "").strip(); pre = re.sub(r"```(?:json)?\s*$", "", pre).strip()
        key = re.sub(r"\s+", "", body)
        c, res = do_call(body); c["tagged"] = tagged; c["duplicate"] = key in seen; calls.append(c); seen.add(key)
        text += (pre + "\n" if pre else "") + "<tool_call>\n" + body.strip() + "\n</tool_call><|im_end|>\n" + im("user", f"<tool_response>\n{res}\n</tool_response>") + "<|im_start|>assistant\n"
        reply += (pre + "\n" if pre else "") + f"<tool_call>{body.strip()}</tool_call><tool_result>{res}</tool_result>"
    else:
        forced = True
    if forced or (calls and not final):
        # cap reached: explicit "now answer" step (the last assistant header is replaced by a user turn)
        text = text[: -len("<|im_start|>assistant\n")] + im("user", NOW_ANSWER) + "<|im_start|>assistant\n"
        out, _ = comp(text); reply += "<now_answer>" + out; final = out.strip()
    return reply, final, calls, dict(now_answer=forced)
def start():
    lg = open(os.path.join(R, "qwen_fair_llama_server.log"), "w")
    p = subprocess.Popen([LS, "-m", GGUF, "-c", "8192", "-ngl", "0", "-t", "6", "--port", str(PORT), "--host", "127.0.0.1", "-np", "1", "-sp"], stdout=lg, stderr=subprocess.STDOUT)
    for _ in range(240):
        try:
            if json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).read()).get("status") == "ok": return p
        except Exception: time.sleep(1)
    p.kill(); raise RuntimeError("server")
def wcsv(fn, rows):
    with open(os.path.join(R, fn), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
with open(os.path.join(R, "qwen_prompts.md"), "w", encoding="utf-8") as f:
    f.write(f"# Exact Qwen prompts ({QN}, local Ollama GGUF blob, llama.cpp llama-server, greedy temp 0)\n\nAll prompts are raw ChatML: `<|im_start|>role\\n...<|im_end|>\\n`.\n\n")
    f.write("## fewshot (our format)\nsystem:\n```\n" + FEWSHOT_SYS + "\n```\nuser: `<question>`; the assistant turn is continued in place; stop at `</tool_call>` / `<|im_end|>`; tool output appended as `<tool_result>...</tool_result>`; up to 3 rounds.\n\n")
    f.write("## native (Qwen chat template)\nsystem:\n```\n" + NATIVE_SYS + "\n```\nuser: `<question>`. A call (tagged, or bare/fenced JSON parsed leniently) is re-rendered as `<tool_call>\\n{json}\\n</tool_call><|im_end|>`, the result is sent as "
            "`<|im_start|>user\\n<tool_response>\\n...\\n</tool_response><|im_end|>` (Qwen2.5's tool-role rendering). Max 2 calls; if the cap is reached a final user turn is added:\n```\n" + NOW_ANSWER + "\n```\n\n")
    f.write("## closed (no tools, reference)\nsystem: `" + CLOSED_SYS + "`, user: `<question>`.\n\nStories: plain prompt as raw text; chat prompt as a ChatML user turn with the closed system prompt.\n")
modes = sys.argv[1:] or ["fewshot", "native", "closed"]
p = start(); t_all = time.time()
try:
    for mode in modes:
        t0 = time.time(); fun, trans = [], []
        items = [("fact", q, qk, ak) for q, qk, ak in FACT] + [("obscure", o["q"], o["qkeys"], o["akeys"]) for o in OBSCURE] + [("math", q, None, a) for q, a in MATH]
        for kind, q, qk, ak in items:
            reply, final, calls, extra = respond(mode, q)
            wiki = [c for c in calls if c["name"] == "wiki_search"]; qtxt = " ".join(str(c["args"].get("query", "")) for c in wiki).lower()
            if kind == "math":
                nums = re.findall(r"-?\d+(?:\.\d+)?", final.replace(",", "")); aok = ak in nums; qok = ""
            else:
                aok = any(k in final.lower() for k in ak); qok = any(k in qtxt for k in qk)
            fun.append(dict(model=QN, mode=mode, kind=kind, item=q, n_calls=len(calls), tagged_calls=sum(c.get("tagged", False) for c in calls), valid_calls=sum(c["valid"] for c in calls),
                            duplicate_calls=sum(c.get("duplicate", False) for c in calls), used_wiki=bool(wiki), query_ok=qok, answer_ok=aok, forced_now_answer=extra["now_answer"], final=final[:300]))
            trans.append(dict(model=QN, mode=mode, kind=kind, q=q, reply=reply)); print(mode, kind, q[:40], "|", final[:100].replace("\n", " "), flush=True)
        if mode == "closed":
            for kind, pr in STORY:
                prompt = pr if kind == "plain" else im("system", CLOSED_SYS) + im("user", "Tell me a short story about a dog who is afraid of the dark.") + "<|im_start|>assistant\n"
                out, _ = comp(prompt, n=120); trans.append(dict(model=QN, mode=mode, kind="story_" + kind, q=pr, reply=out))
        wcsv(f"qwen_fair_{mode}_functional.csv", fun)
        with open(os.path.join(R, f"qwen_fair_{mode}_transcripts.jsonl"), "w", encoding="utf-8") as f:
            for t in trans: f.write(json.dumps(t, ensure_ascii=False) + "\n")
        def agg(k): return [r for r in fun if r["kind"] == k]
        nc = sum(r["n_calls"] for r in fun); vc = sum(r["valid_calls"] for r in fun)
        s = dict(model=QN, mode=mode, decoding="greedy", device="cpu llama.cpp 6 threads", tool_calls=nc, tagged_calls=sum(r["tagged_calls"] for r in fun), valid_json_rate=round(vc / nc, 3) if nc else "",
                 duplicate_calls=sum(r["duplicate_calls"] for r in fun), forced_now_answer=sum(r["forced_now_answer"] for r in fun),
                 fact_used_wiki=sum(r["used_wiki"] for r in agg("fact")), fact_query_ok=sum(bool(r["query_ok"]) for r in agg("fact")), fact_answer_ok=sum(r["answer_ok"] for r in agg("fact")),
                 obscure_used_wiki=sum(r["used_wiki"] for r in agg("obscure")), obscure_query_ok=sum(bool(r["query_ok"]) for r in agg("obscure")), obscure_answer_ok=sum(r["answer_ok"] for r in agg("obscure")),
                 math_ok=sum(r["answer_ok"] for r in agg("math")), math_used_calc=sum(r["n_calls"] > 0 for r in agg("math")), seconds=round(time.time() - t0))
        wcsv(f"qwen_fair_{mode}_summary.csv", [s]); print(json.dumps(s), flush=True)
finally:
    p.kill(); p.wait()
print("total", round(time.time() - t_all))
