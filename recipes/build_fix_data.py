r"""seed_fix1 data (2026-11-24 CST): write -> run -> read error -> patch -> re-run, with a sandboxed python_exec tool (same call format as the calculator).
Python only. Sources:
  (a) MBPP train+validation+prompt (NOT test split) solutions + their asserts. Each solution gets real bug mutations (off-by-one, wrong operator,
      wrong name, missing return, bad indexing, type mix, syntax). Buggy code + tests are RUN in an isolated subprocess (python -I, temp dir, 10 s
      timeout) to capture the REAL traceback/assert failure; the fixed code is re-run and must pass.
  (b) Colby's own Python (colby_code train split) and the 18k alpaca Python answers: functions are COMPILED only (never executed) after a
      syntax mutation, giving real SyntaxError/IndentationError messages and the repair.
No codetest function names (guarded)."""
import os, re, sys, json, random, subprocess, tempfile, ast
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor
OUT = r"D:\seed_v4\seed_fix1\data"; E = "<|endoftext|>"; R = random.Random(20261124)
CODETEST = ["run_length_encode", "second_largest", "is_balanced", "merge_intervals", "word_frequencies", "rotate_clockwise", "caesar_shift", "flatten_deep",
            "digital_root", "chunk_list", "common_prefix", "roman_value", "vowel_counts", "pairs_with_sum", "parse_settings", "moving_average", "snake_to_camel",
            "is_prime", "group_by_length", "spiral_order"]
CT = re.compile(r"def\s+(" + "|".join(CODETEST) + r")\s*\(")
SPEC = {"name": "python_exec", "description": "Run Python code in a sandbox and return stdout/stderr (tracebacks included).", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}
SYS_ = "<|system|>You are a careful Python developer. You can call these tools:\n" + json.dumps(SPEC) + "<|end|>\n"
def run(code, timeout=10):
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "main.py"); open(p, "w", encoding="utf-8").write(code)
        try:
            r = subprocess.run([sys.executable, "-I", p], cwd=td, capture_output=True, text=True, timeout=timeout)
            out = (r.stdout + r.stderr).replace(td, "/sandbox").replace(p, "/sandbox/main.py")
            return r.returncode, out.strip()[-700:]
        except subprocess.TimeoutExpired: return -9, "TimeoutError: execution exceeded 10 seconds"
def clean_tb(out): return re.sub(r'File "[^"]*main\.py"', 'File "main.py"', out)
MUT = []
def mut(f): MUT.append(f); return f
@mut
def off_by_one(c, r):
    ms = list(re.finditer(r"range\(([^()]*(?:\([^()]*\))?[^()]*)\)", c))
    if not ms: return None
    m = r.choice(ms); a = m.group(1)
    new = f"range({a} - 1)" if "," not in a else f"range({a.split(',')[0]}, {a.split(',', 1)[1].strip()} - 1)"
    return c[:m.start()] + new + c[m.end():], f"the loop stops one step early: `{m.group(0)}` was written as `{new}`", "off-by-one in range"
@mut
def wrong_op(c, r):
    ops = [(" + ", " - "), (" - ", " + "), (" * ", " + "), (" < ", " <= "), (" > ", " >= "), (" <= ", " < "), (" >= ", " > "), (" == ", " != "), (" // ", " / ")]
    r.shuffle(ops)
    for a, b in ops:
        idx = [m.start() for m in re.finditer(re.escape(a), c)]
        if idx:
            i = r.choice(idx); line = c[:i].count("\n") + 1
            return c[:i] + b + c[i + len(a):], f"line {line} uses `{b.strip()}` where it needs `{a.strip()}`", f"wrong operator {b.strip()} -> {a.strip()}"
    return None
@mut
def wrong_name(c, r):
    try: t = ast.parse(c)
    except Exception: return None
    names = sorted({n.id for n in ast.walk(t) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and len(n.id) > 1 and n.id not in dir(__builtins__)})
    if not names: return None
    v = r.choice(names); occ = [m.start() for m in re.finditer(r"\b" + re.escape(v) + r"\b", c)]
    if len(occ) < 2: return None
    i = r.choice(occ[1:]); typo = v[:-1] if len(v) > 2 else v + "s"
    return c[:i] + typo + c[i + len(v):], f"`{typo}` is a typo for `{v}`", f"NameError typo {typo} -> {v}"
@mut
def drop_return(c, r):
    ms = [m for m in re.finditer(r"^(\s+)return (.+)$", c, re.M)]
    if not ms: return None
    m = r.choice(ms)
    return c[:m.start()] + f"{m.group(1)}{m.group(2).strip()}" + c[m.end():], f"the value `{m.group(2).strip()[:60]}` is computed but never returned", "missing return"
@mut
def bad_index(c, r):
    ms = list(re.finditer(r"\[(-?\d+)\]", c))
    if not ms: return None
    m = r.choice(ms); k = int(m.group(1)); nk = k + 1 if k >= 0 else k - 1
    return c[:m.start()] + f"[{nk}]" + c[m.end():], f"index `[{nk}]` should be `[{k}]`", "wrong index"
@mut
def type_mix(c, r):
    ms = list(re.finditer(r"return (\w+)$", c, re.M))
    if not ms: return None
    m = r.choice(ms)
    return c[:m.start()] + f"return str({m.group(1)})" + c[m.end():], f"the result is converted to a string with `str(...)`; the tests expect the raw value", "wrong return type"
def syntax_mut(c, r):
    k = r.randrange(4); lines = c.split("\n")
    cand = [i for i, l in enumerate(lines) if l.rstrip().endswith(":") and l.strip()]
    if k == 0 and cand:
        i = r.choice(cand); lines[i] = lines[i].rstrip()[:-1]; return "\n".join(lines), f"line {i + 1} is missing the `:` at the end", "missing colon"
    cand = [i for i, l in enumerate(lines) if l.count("(") >= 1 and l.count("(") == l.count(")") and not l.strip().startswith("#")]
    if k == 1 and cand:
        i = r.choice(cand); j = lines[i].rfind(")"); lines[i] = lines[i][:j] + lines[i][j + 1:]; return "\n".join(lines), f"line {i + 1} is missing a closing `)`", "unclosed paren"
    cand = [i for i in range(1, len(lines)) if lines[i].startswith("        ") and lines[i].strip()]
    if k == 2 and cand:
        i = r.choice(cand); lines[i] = lines[i][2:]; return "\n".join(lines), f"line {i + 1} has the wrong indentation", "bad indentation"
    cand = [i for i, l in enumerate(lines) if re.search(r"\bdef |\bif |\bfor |\bwhile |\breturn ", l)]
    if cand:
        i = r.choice(cand); lines[i] = re.sub(r"\b(def|if|for|while|return) ", lambda m: m.group(1)[:-1] + " ", lines[i], count=1); return "\n".join(lines), f"a keyword is misspelled on line {i + 1}", "misspelled keyword"
    return None
def episode(task, buggy, fixed, tests, tb, why, ok_out, r):
    v = r.randrange(3); fence = lambda x: f"```python\n{x.strip()}\n```"
    call = lambda code: "<tool_call>" + json.dumps({"name": "python_exec", "arguments": {"code": code}}) + "</tool_call>"
    if v < 2:
        s = SYS_ + f"<|user|>{task}<|end|>\n<|assistant|>"
        s += r.choice(["<think>I'll write the function and run the tests.</think>", "<think>Write it, then test it.</think>", ""])
        s += call(buggy.strip() + "\n\n" + tests) + f"<tool_result>{tb}</tool_result>"
        s += f"<think>The run failed. Reading the error: {why}. Fix that and re-run.</think>"
        s += call(fixed.strip() + "\n\n" + tests) + f"<tool_result>{ok_out}</tool_result>"
        s += "The tests pass now. " + fence(fixed) + "<|end|>"
    else:
        s = SYS_ + f"<|user|>This code fails:\n{fence(buggy)}\nError:\n{tb}\nFix it.<|end|>\n<|assistant|><think>{why[0].upper() + why[1:]}.</think>" + fence(fixed) + "<|end|>"
    return s
def mbpp_job(args):
    row, seed = args; r = random.Random(seed); out = []
    code = row["code"].replace("\r\n", "\n").replace("\t", "    ").strip(); tests = "\n".join(row["test_list"]) + "\nprint('all tests passed')"
    if CT.search(code): return out
    rc, ok_out = run(code + "\n\n" + tests)
    if rc != 0 or "all tests passed" not in ok_out: return out
    task = f"{row['text']} Your code should pass these tests:\n" + "\n".join(row["test_list"])
    tried = set()
    for _ in range(14):
        f = r.choice(MUT + [None])
        m = syntax_mut(code, r) if f is None else f(code, r)
        if not m or m[0] in tried or m[0] == code: continue
        tried.add(m[0]); buggy, why, kind = m
        rc2, tb = run(buggy + "\n\n" + tests)
        if rc2 == 0 and "all tests passed" in tb: continue        # mutation did not break anything
        out.append(episode(task, buggy, code, tests, clean_tb(tb), why, ok_out, r))
    return out
if __name__ == "__main__":
    rows = []
    for sp in ("train", "validation", "prompt"): rows += pq.read_table(rf"D:\seed_v4\public_code\mbpp\full\{sp}-00000-of-00001.parquet").to_pylist()
    jobs = [(row, 1000 * rep + i) for rep in range(3) for i, row in enumerate(rows)]
    n = 0
    with open(os.path.join(OUT, "fix_mbpp.txt"), "w", encoding="utf-8", newline="\n") as f, ThreadPoolExecutor(6) as ex:
        for eps in ex.map(mbpp_job, jobs):
            for e in eps: f.write(e + E); n += 1
    print("mbpp run-and-fix episodes", n, flush=True)
    # (b) compile-only syntax repairs on Colby's functions + alpaca answers
    def funcs(text):
        try: t = ast.parse(text)
        except Exception: return []
        L = text.split("\n"); out = []
        for node in t.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.end_lineno - node.lineno < 60:
                src = "\n".join(L[node.lineno - 1 - len(node.decorator_list): node.end_lineno])
                if 3 <= src.count("\n") and not CT.search(src): out.append(src)
        return out
    colby = open(r"D:\seed_v4\colby_code\lang\python\train.txt", encoding="utf-8").read().split(E)
    alp = [re.search(r"```python\n(.*?)\n```", d, re.S) for d in open(r"D:\seed_v4\seed_py1\data\alpaca_py.txt", encoding="utf-8").read().split(E)]
    srcs = [("colby", fn) for d in colby for fn in funcs(re.sub(r"^# file: .*\n", "", d))] + [("alpaca", fn) for m in alp if m for fn in funcs(m.group(1))]
    r = random.Random(7); k = 0
    with open(os.path.join(OUT, "fix_syntax.txt"), "w", encoding="utf-8", newline="\n") as f:
        for who, fn in srcs:
            for _ in range(2 if who == "colby" else 1):
                m = syntax_mut(fn, r)
                if not m: continue
                try: compile(m[0], "main.py", "exec"); continue
                except SyntaxError as e:
                    tb = f'  File "main.py", line {e.lineno}\n    {(e.text or "").rstrip()}\nSyntaxError: {e.msg}' if not isinstance(e, IndentationError) else f'  File "main.py", line {e.lineno}\nIndentationError: {e.msg}'
                try: compile(fn, "main.py", "exec")
                except SyntaxError: continue
                f.write(episode("Fix the syntax of this code and check it compiles.", m[0], fn, "import ast; print('compiles')", tb, m[1], "compiles", r) + E); k += 1
    print("syntax repair episodes", k, "from", len(srcs), "functions", flush=True)
