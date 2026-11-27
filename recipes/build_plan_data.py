"""Planner seed data (2026-11-24 CST, seed_plan1): word problem -> Known values -> ordered Plan steps -> one calculator call per step -> final answer.
Sources: (a) GSM8K train = run4_data/gsm8k.txt FIRST 99% ONLY (the last 1% holds pool3/heldout_gsm8k_tail20.json and is never read; 10 pool test
questions excluded verbatim), each doc rendered in 3 format variants; (b) multi-step template problems (2-4 steps, exact answers from the real
chat100m calculator)."""
import sys, json, random, re, os
sys.path.insert(0, r"D:\tiny100m\src"); import chat100m as C
sys.path.insert(0, r"D:\seed_v4\seed_math1\data"); import build_math_data as M
OUT = r"D:\seed_v4\seed_plan1\data"
HELD = [x["q"] for x in json.load(open(r"D:\seed_v4\pool3\heldout_gsm8k_tail20.json", encoding="utf-8"))]
def calc(e): return C.run_tool(json.dumps({"name": "calculator", "arguments": {"expression": e}}), ".")
EQ = re.compile(r"([\d$][\d\s\.\+\-\*/\(\)\$,x×%]*?[\d\)%])\s*=\s*\$?(-?[\d,]+(?:\.\d+)?)")
NUM = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?")
WORDNUM = {"half": "1/2", "twice": "2 times", "double": "2 times", "triple": "3 times", "quarter": "1/4", "dozen": "12"}
def known(q):
    out = []
    for m in NUM.finditer(q):
        after = re.split(r"[.,;?!]", q[m.end():m.end() + 40])[0].split()[:3]
        before = q[max(0, m.start() - 25):m.start()].split()[-2:]
        out.append((" ".join(before) + " " if m.group().startswith("$") is False and before and before[-1].lower() in ("of", "costs", "cost", "for", "pays", "is", "has", "earns", "every", "each") else "") + m.group() + (" " + " ".join(after) if after else ""))
    for w, v in WORDNUM.items():
        if re.search(r"\b" + w + r"\b", q, re.I): out.append(f"{w} = {v}")
    seen = []; [seen.append(x.strip()) for x in out if x.strip() not in seen]
    return seen[:10]
def norm_expr(e):
    e = e.replace("$", "").replace(",", "").replace("x", "*").replace("×", "*")
    e = re.sub(r"(\d+(?:\.\d+)?)%", r"(\1/100)", e)
    return re.sub(r"\s+", " ", e).strip()
def num_eq(a, b):
    try: return abs(float(a) - float(str(b).replace(",", "").replace("$", ""))) < 1e-6
    except Exception: return False
def fmt_v(v):
    try:
        f = float(v); return str(int(f)) if f == int(f) else v
    except Exception: return v
def gsm_steps(doc):
    m = re.match(r"<\|user\|>(.*?)<\|end\|>\n<\|assistant\|><think>(.*?)</think>The answer is (.*?)\.<\|end\|>", doc.strip(), re.S)
    if not m: return None
    q, th, ans = m.groups(); steps = []
    for l in [l for l in th.split("\n") if l.strip()]:
        mm = list(EQ.finditer(l))
        if not mm: continue
        e = norm_expr(mm[-1].group(1))
        if not re.fullmatch(r"[\d\.\s\+\-\*/\(\)]+", e) or not re.search(r"[\+\-\*/]", e): continue
        v = calc(e)
        if not num_eq(v, mm[-1].group(2)): return None
        desc = (l[:mm[-1].start()] + "(" + e + ")" + l[mm[-1].end():]).strip().rstrip(":").strip(" .") or "Compute the next value"
        steps.append((desc, e, fmt_v(v)))
    if not steps or len(steps) > 8 or not num_eq(steps[-1][2], ans): return None
    return q, steps, fmt_v(steps[-1][2])
def render(r, q, kn, steps, ans, unit=""):
    v = r.randrange(3)
    out = M.system(r) + f"<|user|>{q}<|end|>\n<|assistant|>"
    if v == 0:
        plan = "Known values: " + "; ".join(kn) + ".\nPlan:\n" + "\n".join(f"{i+1}. {s[0]}." for i, s in enumerate(steps))
        out += f"<think>{plan}</think>"
        for i, (d, e, val) in enumerate(steps):
            out += f"<think>Step {i+1}: {d}.</think><tool_call>{json.dumps({'name': 'calculator', 'arguments': {'expression': e}})}</tool_call><tool_result>{val}</tool_result>"
    elif v == 1:
        plan = "Given:\n" + "\n".join("- " + k for k in kn) + f"\nI need {len(steps)} step{'s' if len(steps) > 1 else ''}:\n" + "\n".join(f"{i+1}) {s[0]}" for i, s in enumerate(steps))
        out += f"<think>{plan}</think>"
        for i, (d, e, val) in enumerate(steps):
            out += f"<tool_call>{json.dumps({'name': 'calculator', 'arguments': {'expression': e}})}</tool_call><tool_result>{val}</tool_result>"
    else:
        out += f"<think>Facts: {', '.join(kn)}. Steps: " + " Then ".join(s[0] + "." for s in steps) + "</think>"
        for i, (d, e, val) in enumerate(steps):
            out += f"<think>{d}.</think><tool_call>{json.dumps({'name': 'calculator', 'arguments': {'expression': e}})}</tool_call><tool_result>{val}</tool_result>"
    return out + r.choice([f"The answer is {ans}{unit}.", f"So the answer is {ans}{unit}.", f"Final answer: {ans}{unit}."]) + "<|end|>"
N = M.NAMES; IT = M.ITEMS; P = M.P
def multi(r):
    n = r.choice(N); it = r.choice(IT); f = r.randrange(10)
    if f == 0:
        a, b, c, d = r.randint(10, 90), r.randint(2, 9), r.randint(2, 12), 0; d = r.randint(1, a + b * c - 1)
        q = f"{n} has {a} {it}. {P(n)} buys {b} packs with {c} {it} in each pack, then gives {d} {it} away. How many {it} does {n} have now?"
        kn = [f"{a} {it} at the start", f"{b} packs", f"{c} {it} per pack", f"{d} given away"]
        st = [(f"Find the {it} in the packs", f"{b} * {c}"), (f"Add them to the starting {a}", f"{a} + {{0}}"), (f"Subtract the {d} given away", f"{{1}} - {d}")]
    elif f == 1:
        p, q_, k = r.randint(2, 20), r.randint(2, 15), r.choice([20, 50, 100, 200, 500])
        while p * q_ * 2 > k: k *= 2
        tax = r.choice([5, 10, 20])
        q = f"Notebooks cost {p} dollars each. {n} buys {q_} notebooks and pays {tax} dollars for shipping. {P(n)} pays with {k} dollars. How much change does {P(n).lower()} get?"
        kn = [f"{p} dollars per notebook", f"{q_} notebooks", f"{tax} dollars shipping", f"paid {k} dollars"]
        st = [("Find the cost of the notebooks", f"{p} * {q_}"), ("Add the shipping", f"{{0}} + {tax}"), (f"Subtract the total from {k}", f"{k} - {{1}}")]
    elif f == 2:
        w, h, d = r.randint(8, 30), r.randint(2, 9), r.randint(2, 6); bonus = r.randint(10, 100)
        q = f"{n} earns {w} dollars per hour and works {h} hours a day for {d} days. {P(n)} also gets a {bonus} dollar bonus. How much does {P(n).lower()} earn in total?"
        kn = [f"{w} dollars per hour", f"{h} hours a day", f"{d} days", f"{bonus} dollar bonus"]
        st = [("Find the total hours", f"{h} * {d}"), ("Multiply hours by the hourly pay", f"{{0}} * {w}"), ("Add the bonus", f"{{1}} + {bonus}")]
    elif f == 3:
        a = r.randint(3, 20); m = r.randint(2, 5); s = r.randint(1, 15)
        q = f"Ana has {a} {it}. Ben has {m} times as many as Ana. Cara has {s} fewer than Ben. How many {it} do the three have together?".replace("Ana", n) if False else f"{n} has {a} {it}. Ben has {m} times as many as {n}. Cara has {s} fewer than Ben. How many {it} do the three have together?"
        kn = [f"{n} has {a}", f"Ben has {m} times as many", f"Cara has {s} fewer than Ben"]
        st = [("Find how many Ben has", f"{a} * {m}"), ("Find how many Cara has", f"{{0}} - {s}"), ("Add all three", f"{a} + {{0}} + {{1}}")]
    elif f == 4:
        t = r.choice([40, 60, 80, 100, 120, 200, 240, 300, 400]); p1 = r.choice([10, 20, 25, 50]); p2 = r.choice([10, 20, 25, 50])
        q = f"A store has {t} {it}. It sells {p1}% of them on Monday and {p2}% of the remaining {it} on Tuesday. How many {it} are left?"
        kn = [f"{t} {it}", f"{p1}% sold Monday", f"{p2}% of the rest sold Tuesday"]
        st = [("Find how many were sold on Monday", f"{t} * {p1}/100"), ("Find how many remained after Monday", f"{t} - {{0}}"), ("Find how many were sold on Tuesday", f"{{1}} * {p2}/100"), ("Subtract Tuesday's sales", f"{{1}} - {{2}}")]
    elif f == 5:
        sp, h1, h2 = r.randint(20, 70), r.randint(1, 5), r.randint(1, 4); sp2 = r.randint(20, 70)
        q = f"A car drives {sp} miles per hour for {h1} hours, then {sp2} miles per hour for {h2} hours. How many miles does it drive in total?"
        kn = [f"{sp} mph for {h1} hours", f"{sp2} mph for {h2} hours"]
        st = [("Find the first distance", f"{sp} * {h1}"), ("Find the second distance", f"{sp2} * {h2}"), ("Add both distances", "{0} + {1}")]
    elif f == 6:
        k = r.randint(3, 8); per = r.randint(2, 9); price = r.randint(2, 15); sold = r.randint(1, k * per)
        q = f"{n} bakes {k} trays of muffins with {per} muffins on each tray. {P(n)} sells {sold} muffins for {price} dollars each. How many muffins are left, and how much money did {P(n).lower()} make? Give the money."
        kn = [f"{k} trays", f"{per} muffins per tray", f"{sold} sold", f"{price} dollars each"]
        st = [("Find the total muffins", f"{k} * {per}"), ("Find the muffins left", f"{{0}} - {sold}"), ("Find the money made", f"{sold} * {price}")]
    elif f == 7:
        L, W, c = r.randint(3, 30), r.randint(3, 30), r.randint(2, 12)
        q = f"A garden is {L} feet long and {W} feet wide. Fencing costs {c} dollars per foot. How much does it cost to fence the whole garden?"
        kn = [f"{L} feet long", f"{W} feet wide", f"{c} dollars per foot"]
        st = [("Add the length and width", f"{L} + {W}"), ("Double it for the perimeter", "2 * {0}"), ("Multiply by the price per foot", f"{{1}} * {c}")]
    elif f == 8:
        g = r.randint(2, 6); s = r.randint(10, 30) * g; x = r.randint(1, 9)
        q = f"{n} has {s} {it} and splits them equally into {g} bags. Then {P(n).lower()} adds {x} more to each bag. How many {it} are in each bag?"
        kn = [f"{s} {it}", f"{g} bags", f"{x} more per bag"]
        st = [("Divide the items among the bags", f"{s} / {g}"), ("Add the extra items", f"{{0}} + {x}")]
    else:
        a, b = r.randint(5, 40), r.randint(2, 12); y = r.randint(2, 20)
        q = f"{n} is {a} years old. {P(n)} is {b} years older than {'her' if P(n) == 'She' else 'his'} cousin. How old will the cousin be in {y} years?"
        kn = [f"{n} is {a}", f"{b} years older than the cousin", f"in {y} years"]
        st = [("Find the cousin's age now", f"{a} - {b}"), (f"Add {y} years", f"{{0}} + {y}")]
    vals = []; steps = []
    for d, e in st:
        e = e.format(*vals); v = fmt_v(calc(e))
        if not re.fullmatch(r"-?\d+", v) or int(v) < 0: return None
        vals.append(v); steps.append((d, e, v))
    return q, kn, steps, vals[-1]
if __name__ == "__main__":
    r = random.Random(20261124 + 7)
    p = r"D:\seed_v4\run4_data\gsm8k.txt"; size = os.path.getsize(p)
    raw = open(p, "rb").read(int(size * 0.99)).decode("utf-8", "ignore").split("<|endoftext|>")[:-1]
    ng = nt = 0
    with open(os.path.join(OUT, "plan_gsm.txt"), "w", encoding="utf-8", newline="\n") as f:
        for d in raw:
            g = gsm_steps(d)
            if not g: continue
            q, steps, ans = g
            if q in M.TEST or any(q[:60] in h or h[:60] in q for h in HELD): continue
            kn = known(q)
            for _ in range(3): f.write(render(r, q, kn, steps, ans) + "<|endoftext|>"); ng += 1
    with open(os.path.join(OUT, "plan_synth.txt"), "w", encoding="utf-8", newline="\n") as f:
        while nt < 120000:
            m = multi(r)
            if m: f.write(render(r, m[0], m[1], m[2], m[3]) + "<|endoftext|>"); nt += 1
    print("gsm docs written", ng, "(3 variants each) of", len(raw), "| synth", nt)
