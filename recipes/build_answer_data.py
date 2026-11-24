"""Answer-from-results data for seed_answer1 (2026-11-24): question -> real wiki_search (same bank + same formatting as the
chat100m / test harness) -> short final answer = the first sentence of the MATCHING article's lead (not a medal, house or
monument that merely shares the name). If the right article is not in the results, the answer says so.
The 20 test entities are excluded. Run4's held-out tails are never read (this data comes only from the Wikipedia bank)."""
import sys, re, json, random, time
from multiprocessing import Pool
WIKI = r"D:\wiki_memory\part\wiki_leads_en_v1"
EXCL = ["einstein", "marie curie", "eiffel", "amazon river", "isaac newton", "everest", "shakespeare", "python (programming", "da vinci",
        "great wall", "abraham lincoln", "photosynthesis", "mars", "nelson mandela", "tokyo", "charles darwin", "beatles", "moon", "picasso", "mozart"]
SPECS3 = ('{"name": "wiki_search", "description": "Search Wikipedia article introductions and return the best matches", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "what to look up"}}, "required": ["query"]}}',
          '{"name": "calculator", "description": "Evaluate an arithmetic expression exactly", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "e.g. 12*(3+4)"}}, "required": ["expression"]}}',
          '{"name": "read_file", "description": "Read a text file from the user\'s workspace", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "file name"}}, "required": ["path"]}}')
THINK = ["I should check Wikipedia before answering.", "This is a factual question. Let me search Wikipedia for {t}.",
         "I am not sure about the details of {t}, so I will look it up.", "The user is asking about {t}. I should search Wikipedia for accurate facts."]
_wm = None
def init():
    global _wm
    sys.path.insert(0, WIKI); import wiki_memory; _wm = wiki_memory.default_memory()
def fmt(res):
    out = []
    for i, (t, x, _) in enumerate(res):
        x = x if len(x) <= 320 else x[:320].rsplit(" ", 1)[0] + " ..."
        out.append(f"{i+1}. {t}: {x}")
    return "\n".join(out) if out else "No results."
SENT = re.compile(r"(?<=[a-z0-9\)\]\"'][.!?])\s+(?=[A-Z])")
def first_sentence(x):
    x = re.sub(r"\s+", " ", x).strip()
    for m in SENT.finditer(x):
        if m.start() >= 30:
            return x[:m.start()]
    return x[:300].rsplit(" ", 1)[0] if len(x) > 300 else x
def norm(s): return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
def make(i):
    r = random.Random(1000 + i)
    if r.random() < 0.5 and _wm.vital and _wm.vital.n:
        title, lead = _wm.vital.docs[r.randrange(_wm.vital.n)]
    else:
        title, lead = _wm.get_id(r.randrange(int(_wm.manifest["entry_count"])))
    low = title.lower()
    if any(e in low for e in EXCL) or len(lead) < 80 or "(disambiguation)" in low or low.startswith("list of"):
        return None
    person = bool(re.search(r"\((born|c\. |\d{1,2} \w+ \d{3,4}|\w+ \d{1,2}, \d{3,4}|\d{3,4} ?[–-])", lead[:250]))
    q = r.choice([f"Who was {title}?", f"Who is {title}?", f"Tell me about {title}."] if person else [f"What is {title}?", f"What is {title}?", f"Tell me about {title}."])
    query = title if r.random() < 0.8 else q.rstrip("?.").replace("Tell me about ", "")
    res = _wm.search(query, 3)
    hit = next((k for k, (t, x, _) in enumerate(res) if norm(t) == norm(title)), None)
    sys_ = "<|system|>You are a helpful assistant. You can call these tools:\n" + "\n".join(SPECS3 if r.random() < 0.6 else SPECS3[:2][::-1]) + "<|end|>\n"
    think = r.choice(THINK).format(t=title)
    if hit is None:
        ans = f"The search results do not mention {title}, so I could not find a reliable answer."
    else:
        ans = first_sentence(res[hit][1])
    call = json.dumps({"name": "wiki_search", "arguments": {"query": query}}, ensure_ascii=False)
    return (sys_ + f"<|user|>{q}<|end|>\n<|assistant|><think>{think}</think><tool_call>{call}</tool_call>"
            f"<tool_result>{fmt(res)}</tool_result>{ans}<|end|>"), hit is not None
if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
    t0 = time.time(); n_ok = n_miss = 0
    with Pool(7, initializer=init) as p, open(r"D:\seed_v4\seed_answer1\data\synth_answer.txt", "w", encoding="utf-8", newline="\n") as f:
        for k, out in enumerate(p.imap_unordered(make, range(N), chunksize=64)):
            if out is None: continue
            doc, hit = out
            if not hit and random.random() < 0.6: continue   # keep 'not found' answers at a modest share
            f.write(doc + "<|endoftext|>"); n_ok += hit; n_miss += (not hit)
            if k % 5000 == 0: print(k, n_ok, n_miss, round(time.time() - t0), flush=True)
    print("done", n_ok, n_miss, round(time.time() - t0))

