r"""seed_py1 data (2026-11-24 CST): Python ONLY.
  public_py.txt   : CPython 3.12 stdlib + Python sources of installed packages (Seed .venv site-packages, qwen_env transformers/tokenizers/accelerate);
                    deduped, minified/huge files dropped, files defining any codetest function name dropped.
  mbpp_inst.txt   : MBPP full train+validation+prompt splits (NOT the MBPP test split, which is kept free as an extra eval) in chat + docstring formats.
  alpaca_py.txt   : iamtarun/python_code_instructions_18k_alpaca in chat format (codetest-name filter applied).
  colby_py.txt    : Colby's own Python, train split from D:\seed_v4\colby_code (the held-out 10% is never used).
HumanEval is NOT used (kept as a possible external eval). The 20-problem codetest is original and grep-guarded."""
import os, re, json, hashlib, random, sys
import pyarrow.parquet as pq
exec(open(r"D:\seed_v4\colby_code\build_code_corpus.py", encoding="utf-8").read().split("CT_RE =")[0].split("from tokenizers")[0])
CODETEST = ["run_length_encode", "second_largest", "is_balanced", "merge_intervals", "word_frequencies", "rotate_clockwise", "caesar_shift", "flatten_deep",
            "digital_root", "chunk_list", "common_prefix", "roman_value", "vowel_counts", "pairs_with_sum", "parse_settings", "moving_average", "snake_to_camel",
            "is_prime", "group_by_length", "spiral_order"]
CT_RE = re.compile(r"def\s+(" + "|".join(CODETEST) + r")\s*\(")
OUT = r"D:\seed_v4\seed_py1\data"; E = "<|endoftext|>"; r = random.Random(1124)
SYS_ = "<|system|>You are a helpful coding assistant. Write correct, clean Python.<|end|>\n"
roots = [r"C:\Users\mcgar\AppData\Local\Programs\Python\Python312\Lib", r"C:\Users\mcgar\Desktop\Seed\.venv\Lib\site-packages",
         r"D:\seed_v4\qwen_env\Lib\site-packages\transformers", r"D:\seed_v4\qwen_env\Lib\site-packages\accelerate"]
seen = set(); n = b = drop = 0
with open(os.path.join(OUT, "public_py.txt"), "w", encoding="utf-8", newline="\n") as f:
    for root in roots:
        for dp, ds, fs in os.walk(root):
            ds[:] = [x for x in ds if x != "__pycache__" and not (root.endswith("Lib") and x == "site-packages")]
            for fn in fs:
                if not fn.endswith(".py"): continue
                p = os.path.join(dp, fn)
                try: t = open(p, encoding="utf-8").read()
                except Exception: drop += 1; continue
                if not t.strip() or len(t) > 200_000: drop += 1; continue
                ls = t.split("\n")
                if max(map(len, ls)) > 1000 or len(t) / len(ls) > 150 or CT_RE.search(t): drop += 1; continue
                h = hashlib.sha1(re.sub(r"\s+", "", t).encode()).hexdigest()
                if h in seen: drop += 1; continue
                seen.add(h); rel = "/".join(p.replace("\\", "/").split("/")[-3:])
                f.write(f"# file: {rel}\n{t}{E}"); n += 1; b += len(t)
print("public_py files", n, "MB", round(b / 2**20, 1), "dropped", drop)
def mbpp_rows():
    for sp in ("train", "validation", "prompt"):
        for row in pq.read_table(rf"D:\seed_v4\public_code\mbpp\full\{sp}-00000-of-00001.parquet").to_pylist(): yield row
k = 0
with open(os.path.join(OUT, "mbpp_inst.txt"), "w", encoding="utf-8", newline="\n") as f:
    for row in mbpp_rows():
        code = row["code"].replace("\r\n", "\n").replace("\t", "    "); tests = "\n".join(row["test_list"])
        if CT_RE.search(code): continue
        for v in range(2):
            if v == 0:
                f.write(SYS_ + f"<|user|>{row['text']} Your code should pass these tests:\n{tests}<|end|>\n<|assistant|>```python\n{code.strip()}\n```<|end|>{E}")
            else:
                m = re.search(r"def\s+\w+\s*\([^)]*\)\s*:", code)
                if not m: continue
                doc = f'    """{row["text"]}\n    """\n'
                f.write(code[:m.end()] + "\n" + doc + code[m.end():].lstrip("\n") + "\n\n" + tests + f"\n{E}")
            k += 1
print("mbpp docs", k)
k = 0
with open(os.path.join(OUT, "alpaca_py.txt"), "w", encoding="utf-8", newline="\n") as f:
    for row in pq.read_table(r"D:\seed_v4\public_code\python_code_instructions_18k_alpaca\data\train-00000-of-00001-8b6e212f3e1ece96.parquet").to_pylist():
        out = (row.get("output") or "").replace("\r\n", "\n")
        if not out.strip() or CT_RE.search(out): continue
        q = row["instruction"] + (("\n\nInput: " + row["input"]) if (row.get("input") or "").strip() else "")
        f.write(SYS_ + f"<|user|>{q}<|end|>\n<|assistant|>```python\n{out.strip()}\n```<|end|>{E}"); k += 1
print("alpaca docs", k)
src = open(r"D:\seed_v4\colby_code\lang\python\train.txt", encoding="utf-8").read()
open(os.path.join(OUT, "colby_py.txt"), "w", encoding="utf-8", newline="\n").write(src); print("colby_py MB", round(len(src.encode()) / 2**20, 2))
