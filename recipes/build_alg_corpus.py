"""Build the seed_alg1 hand-made algorithmic-reasoning corpus.
- Holds out ~15 entire families (zero problem-family overlap) -> alg_heldout.txt
- Trains on the rest + verified cross-language bridges + debugging docs -> alg_reason_train.txt
- Writes alg_corpus_summary.json with family/category/language/diversity stats.
Usage: python build_corpus.py [docs_per_family]"""
import json, os, random, sys, collections, re, hashlib
sys.path.insert(0, os.path.dirname(__file__))
import core, fam_arrays, fam_strings_math, fam_dp_greedy, fam_graph_misc, render, bridges, fam_debug
PER = int(sys.argv[1]) if len(sys.argv) > 1 else 480
DATA = r"D:\seed_v4\seed_alg1\data"
os.makedirs(DATA, exist_ok=True)
rng = random.Random(20261126)
fams = list(core.FAM)
# ---- choose ~15 whole families to hold out, spread across categories ----
by_cat = collections.defaultdict(list)
for f in fams: by_cat[f["cat"]].append(f)
held_names = set()
for cat, fs in sorted(by_cat.items()):
    k = max(1, round(len(fs) * 0.13))
    for f in random.Random(hash(cat) & 0xffff).sample(fs, min(k, len(fs))): held_names.add(f["name"])
# cap held-out near 15
held_names = set(sorted(held_names)[:16])
train_fams = [f for f in fams if f["name"] not in held_names]
held_fams = [f for f in fams if f["name"] in held_names]
def write_family_docs(fh, famlist, per, tag):
    n = 0; lang_counter = collections.Counter(); fam_counter = collections.Counter()
    for f in famlist:
        seen = set()
        for i in range(per):
            r = random.Random(hash((f["name"], tag, i)) & 0x7fffffff)
            try: d = render.render_family(r, f)
            except Exception: continue
            h = hashlib.md5(d.encode()).hexdigest()
            if h in seen: continue
            seen.add(h); fh.write(d); n += 1; fam_counter[f["name"]] += 1
            for lg, pat in (("C","```c"),("JavaScript","```javascript"),("PowerShell","```powershell"),("Bash","```bash"),("Python","```python")):
                if pat in d: lang_counter[lg] += 1
    return n, lang_counter, fam_counter
stats = {}
# ---- TRAIN file ----
train_path = os.path.join(DATA, "alg_reason_train.txt")
with open(train_path, "w", encoding="utf-8") as fh:
    nfam, langc, famc = write_family_docs(fh, train_fams, PER, "train")
    # verified cross-language bridges
    bv = json.load(open(os.path.join(os.path.dirname(__file__), "bridges_verified.json")))
    random.Random(1).shuffle(bv)
    nb = 0
    for item in bv:
        r = random.Random(nb * 7 + 1)
        fh.write(render.render_bridge(r, item)); nb += 1
        langc[{"c":"C","js":"JavaScript","ps":"PowerShell","sh":"Bash","python":"Python"}[item["lang"]]] += 1
    # translation pairs (same key, two different languages, same input)
    by_key = collections.defaultdict(list)
    for item in bv: by_key[item["key"]].append(item)
    ntr = 0
    for key, items in by_key.items():
        for _ in range(110):
            a = random.Random(ntr).choice(items)
            others = [x for x in items if x["lang"] != a["lang"] and x["v"] == a["v"]]
            if not others: continue
            b = random.Random(ntr + 99).choice(others)
            fh.write(render.render_bridge_translate(random.Random(ntr), (a, b))); ntr += 1
    # debugging docs
    ndbg = 0
    for b in fam_debug.BUGS:
        for i in range(420):
            r = random.Random(hash((b["name"], i)) & 0x7fffffff)
            try: inst = fam_debug.instance(b, r)
            except Exception: continue
            fh.write(render.render_debug(r, b, inst)); ndbg += 1; langc["Python"] += 1
stats["train"] = dict(family_docs=nfam, bridge_docs=nb, translate_docs=ntr, debug_docs=ndbg, families=len(train_fams))
# ---- HELDOUT file (entire families unseen in training) ----
held_path = os.path.join(DATA, "alg_heldout.txt")
with open(held_path, "w", encoding="utf-8") as fh:
    nh, langh, famh = write_family_docs(fh, held_fams, min(PER, 220), "held")
stats["heldout"] = dict(family_docs=nh, families=len(held_fams), held_family_names=sorted(held_names))
# ---- diversity check on the TRAIN file ----
def diversity(path):
    prefixes = collections.Counter(); lines = collections.Counter(); ndoc = 0; total_lines = 0
    data = open(path, encoding="utf-8").read()
    for d in data.split("<|endoftext|>"):
        d = d.strip()
        if not d: continue
        body = d
        if body.startswith("<|system|>"):
            body = body.split("<|end|>", 1)[1] if "<|end|>" in body else body
        body = body.replace("<|user|>", "").lstrip()
        ndoc += 1; prefixes[body[:100]] += 1
        for ln in d.splitlines():
            ln = ln.strip()
            if len(ln) > 12: lines[ln] += 1; total_lines += 1
    top_pref = prefixes.most_common(1)[0]
    top_line = lines.most_common(3)
    return dict(docs=ndoc, unique_80char_prefixes=len(prefixes), unique_prefix_ratio=round(len(prefixes)/ndoc, 4),
        most_common_prefix_count=top_pref[1], most_common_prefix_share=round(top_pref[1]/ndoc, 5),
        top_lines=[(c, round(c/total_lines, 5), ln[:60]) for ln, c in top_line], size_mb=round(os.path.getsize(path)/1e6, 2))
# merge language counters
all_lang = collections.Counter(langc)
stats["languages_in_train_docs"] = dict(all_lang)
stats["categories"] = {c: len(fs) for c, fs in sorted(by_cat.items())}
stats["total_families"] = len(fams)
stats["diversity_train"] = diversity(train_path)
stats["diversity_heldout"] = diversity(held_path)
stats["train_bytes"] = os.path.getsize(train_path)
stats["heldout_bytes"] = os.path.getsize(held_path)
json.dump(stats, open(os.path.join(DATA, "alg_corpus_summary.json"), "w"), indent=2)
print(json.dumps(stats, indent=2))


