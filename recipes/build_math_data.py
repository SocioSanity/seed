"""Math seed data (2026-11-24 CST): word problems -> short plan in <think> -> calculator call(s) -> exact result -> final sentence with the number.
Sources: (a) template families (exact answers, real chat100m calculator output), (b) GSM8K train (run4_data/gsm8k.txt, FIRST 99% ONLY) converted to
step-by-step calculator calls (<=3 steps, last value must equal the stated answer). The 10 test questions are excluded verbatim; the gsm8k tail
(last 1%) is never read -> usable as a held-out math test."""
import sys, json, random, re, os
sys.path.insert(0, r"D:\tiny100m\src"); import chat100m as C
SPECS = [json.dumps(v) for v in C.SPECS.values()]
TEST = {"Compute 642 + 135", "What is 347 * 29?", "Compute 1250 - 487", "What is 144 / 12?", "Compute 23 * 17 + 9",
        "Tom has 12 apples. He gives 5 to his sister and buys 8 more. How many apples does Tom have now?",
        "A shop sells pencils for 3 dollars each. Sara buys 7 pencils and pays with a 50 dollar bill. How much change does she get?",
        "There are 4 boxes with 18 books in each box. How many books are there in total?",
        "Mike runs 5 miles every day for 2 weeks. How many miles does he run in total?",
        "A baker makes 120 cookies and sells 3/4 of them. How many cookies are left?"}
def calc(e): return C.run_tool(json.dumps({"name": "calculator", "arguments": {"expression": e}}), ".")
def system(r):
    k = r.random()
    s = SPECS if k < 0.6 else ([SPECS[1]] if k < 0.8 else [SPECS[1], SPECS[0]])
    return "<|system|>You are a helpful assistant. You can call these tools:\n" + "\n".join(s) + "<|end|>\n"
NAMES = ["Tom", "Sara", "Mike", "Anna", "Ben", "Lily", "Omar", "Maya", "Jack", "Zoe", "Leo", "Nina", "Sam", "Ivy", "Raj", "Emma", "Kai", "Lucy", "Noah", "Mia", "Eli", "Ava", "Dan", "Rosa"]
PRON = {"Sara": "She", "Anna": "She", "Lily": "She", "Maya": "She", "Zoe": "She", "Nina": "She", "Ivy": "She", "Emma": "She", "Lucy": "She", "Mia": "She", "Ava": "She", "Rosa": "She"}
ITEMS = ["apples", "marbles", "stickers", "cards", "pencils", "books", "candies", "coins", "stamps", "shells", "toy cars", "cookies", "balloons", "eggs"]
REL = ["his sister", "her brother", "a friend", "his mom", "her teacher", "a neighbor", "his cousin", "her dad"]
def P(n): return PRON.get(n, "He")
def fam(r):
    n = r.choice(NAMES); it = r.choice(ITEMS); f = r.randrange(22)
    if f == 0:
        a, b, c = r.randint(8, 90), 0, r.randint(2, 40); b = r.randint(1, a - 1)
        return (f"{n} has {a} {it}. {P(n)} gives {b} to {r.choice(REL)} and buys {c} more. How many {it} does {n} have now?",
                f"{n} starts with {a}, gives away {b}, then gets {c} more: {a} - {b} + {c}.", f"{a} - {b} + {c}", f"{n} has {{v}} {it} now.")
    if f == 1:
        p, q = r.randint(2, 15), r.randint(2, 12); bill = r.choice([20, 50, 100, 200]);
        while p * q >= bill: q -= 1
        if q < 1: q = 1
        if p * q >= bill: bill = 200
        thing = r.choice(["pencils", "notebooks", "pens", "erasers", "sandwiches", "tickets", "candles"])
        return (f"A shop sells {thing} for {p} dollars each. {n} buys {q} {thing} and pays with a {bill} dollar bill. How much change does {P(n).lower()} get?",
                f"The cost is {p} * {q}. The change is {bill} minus the cost: {bill} - {p} * {q}.", f"{bill} - {p} * {q}", f"{n} gets {{v}} dollars in change.")
    if f == 2:
        g, k = r.randint(2, 30), r.randint(3, 60); box = r.choice(["boxes", "bags", "shelves", "crates", "baskets", "rows"])
        return (f"There are {g} {box} with {k} {it} in each. How many {it} are there in total?", f"{g} {box} with {k} each is {g} * {k}.", f"{g} * {k}", f"There are {{v}} {it} in total.")
    if f == 3:
        m, w = r.randint(1, 15), r.randint(2, 8); act = r.choice([("runs", "miles"), ("swims", "laps"), ("reads", "pages"), ("walks", "kilometers"), ("bikes", "miles")])
        return (f"{n} {act[0]} {m} {act[1]} every day for {w} weeks. How many {act[1]} does {P(n).lower()} {act[0][:-1]} in total?",
                f"A week has 7 days, so {w} weeks is 7 * {w} days. Total: {m} * 7 * {w}.", f"{m} * 7 * {w}", f"{n} {act[0]} {{v}} {act[1]} in total.")
    if f == 4:
        a, b = r.choice([(1, 2), (1, 4), (3, 4), (1, 3), (2, 3), (1, 5), (2, 5), (3, 5)]); t = b * r.randint(4, 60)
        who = r.choice(["A baker", "A farmer", "A shop", n]); obj = r.choice(["cookies", "muffins", "eggs", "apples", "cupcakes", "flowers"])
        return (f"{who} makes {t} {obj} and sells {a}/{b} of them. How many {obj} are left?", f"Sold: {t} * {a}/{b}. Left: {t} - {t} * {a}/{b}.", f"{t} - {t} * {a}/{b}", f"{{v}} {obj} are left.")
    if f == 5:
        k = r.randint(2, 12); t = k * r.randint(2, 40)
        return (f"{n} shares {t} {it} equally among {k} friends. How many {it} does each friend get?", f"Sharing equally means dividing: {t} / {k}.", f"{t} / {k}", f"Each friend gets {{v}} {it}.")
    if f == 6:
        a, b = r.randint(2, 12), r.randint(4, 24); c = r.randint(1, a * b - 1)
        return (f"{n} buys {a} packs of {it} with {b} in each pack and gives away {c}. How many {it} are left?", f"Total is {a} * {b}, minus the {c} given away: {a} * {b} - {c}.", f"{a} * {b} - {c}", f"{n} has {{v}} {it} left.")
    if f == 7:
        s, w = r.randint(3, 50), r.randint(2, 20); x = r.randint(1, s * w - 1)
        return (f"{n} saves {s} dollars every week for {w} weeks and then spends {x} dollars on a gift. How much money is left?", f"Saved {s} * {w}, then spent {x}: {s} * {w} - {x}.", f"{s} * {w} - {x}", f"{n} has {{v}} dollars left.")
    if f == 8:
        a, b = r.randint(4, 40), r.randint(2, 30); m = r.choice(["2", "3"])
        if r.random() < 0.5:
            return (f"{n} is {a} years old. {r.choice(NAMES)} is {b} years older. What is the sum of their ages?", f"The other person is {a} + {b}. Sum: {a} + ({a} + {b}).", f"{a} + ({a} + {b})", f"The sum of their ages is {{v}}.")
        return (f"{n} is {a} years old. {'Her' if P(n) == 'She' else 'His'} father is {m} times as old. How old is the father?",
                f"{m} times {a}: {a} * {m}.", f"{a} * {m}", f"The father is {{v}} years old.")
    if f == 9:
        t = r.choice([20, 40, 50, 60, 80, 100, 120, 150, 200, 240, 300, 400, 500]); p = r.choice([10, 20, 25, 30, 40, 50, 75])
        return (f"{n} has {t} dollars and spends {p}% of it. How much money does {P(n).lower()} have left?", f"{p}% of {t} is {t} * {p}/100. Left: {t} - {t} * {p}/100.", f"{t} - {t} * {p}/100", f"{n} has {{v}} dollars left.")
    if f == 10:
        a, b, c = r.randint(2, 60), r.randint(2, 60), r.randint(2, 60)
        return (f"{n} buys a book for {a} dollars, a pen for {b} dollars and a bag for {c} dollars. How much does {P(n).lower()} spend in total?", f"Add the three prices: {a} + {b} + {c}.", f"{a} + {b} + {c}", f"{n} spends {{v}} dollars in total.")
    if f == 11:
        a, p, b, q = r.randint(1, 9), r.randint(3, 40), r.randint(1, 9), r.randint(3, 60)
        return (f"{n} buys {a} shirts at {p} dollars each and {b} pairs of pants at {q} dollars each. What is the total cost?", f"Shirts cost {a} * {p}, pants cost {b} * {q}: {a} * {p} + {b} * {q}.", f"{a} * {p} + {b} * {q}", f"The total cost is {{v}} dollars.")
    if f == 12:
        d = r.randint(2, 30); rr = r.randint(5, 50); N = d * rr
        return (f"A book has {N} pages. {n} reads {rr} pages a day. How many days will it take to finish the book?", f"Divide the pages by pages per day: {N} / {rr}.", f"{N} / {rr}", f"It will take {{v}} days.")
    if f == 13:
        x, y = r.randint(3, 80), r.randint(1, 40); m2 = r.choice(NAMES)
        return (f"{n} has {x} {it}. {m2} has {y} more {it} than {n}. How many {it} do they have together?", f"{m2} has {x} + {y}. Together: {x} + ({x} + {y}).", f"{x} + ({x} + {y})", f"Together they have {{v}} {it}.")
    if f == 14:
        a, b, c = r.randint(2, 999), r.randint(2, 999), r.randint(2, 99); op = r.choice(["+", "-", "*"]); op2 = r.choice(["+", "-", "*"])
        e = r.choice([f"{a} {op} {b}", f"{a} {op} {c} {op2} {b}", f"{c} * {r.randint(2, 99)} + {r.randint(1, 99)}", f"({a} + {b}) * {c}"])
        q = r.choice([f"Compute {e}", f"What is {e}?", f"Calculate {e}.", f"What is the value of {e}?"])
        return (q, f"I will use the calculator for {e}.", e, f"{e} = {{v}}.")
    if f == 15:
        k = r.randint(2, 40); t = k * r.randint(2, 99)
        return (r.choice([f"What is {t} / {k}?", f"Compute {t} / {k}", f"Divide {t} by {k}."]), f"I will use the calculator for {t} / {k}.", f"{t} / {k}", f"{t} / {k} = {{v}}.")
    if f == 16:
        k = r.choice([3, 4]); xs = [r.randint(50, 100) for _ in range(k)]; s = sum(xs)
        while s % k: xs[0] += 1; s = sum(xs)
        return (f"{n} scored {', '.join(map(str, xs[:-1]))} and {xs[-1]} on {k} tests. What is the average score?", f"Average = sum / {k}: ({' + '.join(map(str, xs))}) / {k}.", f"({' + '.join(map(str, xs))}) / {k}", f"The average score is {{v}}.")
    if f == 17:
        a, b = r.randint(2, 50), r.randint(2, 50)
        if r.random() < 0.5:
            return (f"A rectangle is {a} meters long and {b} meters wide. What is its area?", f"Area = length * width: {a} * {b}.", f"{a} * {b}", f"The area is {{v}} square meters.")
        return (f"A rectangle is {a} meters long and {b} meters wide. What is its perimeter?", f"Perimeter = 2 * (length + width): 2 * ({a} + {b}).", f"2 * ({a} + {b})", f"The perimeter is {{v}} meters.")
    if f == 18:
        h, m = r.randint(1, 4), r.randint(1, 59)
        return (f"A movie lasts {h} hours and {m} minutes. How many minutes is that?", f"Each hour is 60 minutes: {h} * 60 + {m}.", f"{h} * 60 + {m}", f"The movie is {{v}} minutes long.")
    if f == 19:
        p, h, d = r.randint(8, 30), r.randint(2, 10), r.randint(2, 7); w = r.randint(1, 3)
        return (f"{w} workers are each paid {p} dollars per hour. They each work {h} hours a day for {d} days. What is the total cost?", f"Total hours {w} * {h} * {d}, times {p} per hour: {w} * {h} * {d} * {p}.", f"{w} * {h} * {d} * {p}", f"The total cost is {{v}} dollars.")
    if f == 20:
        a, b = r.randint(20, 500), r.randint(2, 19); c = r.randint(1, 30)
        return (f"A school has {a} students. {b} more students join in the fall and {c} leave in the spring. How many students are there now?", f"{a} + {b} - {c}.", f"{a} + {b} - {c}", f"There are {{v}} students now.")
    k, t = r.randint(2, 9), r.randint(2, 9); per = r.randint(2, 25)
    return (f"Each of {k} children has {t} bags with {per} {it} in each bag. How many {it} are there altogether?", f"{k} children * {t} bags * {per} {it}: {k} * {t} * {per}.", f"{k} * {t} * {per}", f"There are {{v}} {it} altogether.")
def synth(r):
    q, think, e, fin = fam(r)
    if q in TEST: return None
    v = calc(e)
    if not re.fullmatch(r"-?\d+(\.\d+)?", v): return None
    if "." in v and float(v) != int(float(v)): return None
    call = json.dumps({"name": "calculator", "arguments": {"expression": e}})
    return system(r) + f"<|user|>{q}<|end|>\n<|assistant|><think>{think}</think><tool_call>{call}</tool_call><tool_result>{v}</tool_result>{fin.format(v=v)}<|end|>"
EQ = re.compile(r"([\d$][\d\s\.\+\-\*/\(\)\$,x×]*?[\d\)])\s*=\s*\$?(-?[\d,]+(?:\.\d+)?)")
def from_gsm(doc, r):
    m = re.match(r"<\|user\|>(.*?)<\|end\|>\n<\|assistant\|><think>(.*?)</think>The answer is (.*?)\.<\|end\|>", doc.strip(), re.S)
    if not m: return None
    q, th, ans = m.groups(); lines = [l for l in th.split("\n") if l.strip()]
    steps = []
    for l in lines:
        mm = list(EQ.finditer(l))
        if not mm: continue
        e = mm[-1].group(1).replace("$", "").replace(",", "").replace("x", "*").replace("×", "*").strip()
        if not re.fullmatch(r"[\d\.\s\+\-\*/\(\)]+", e) or not re.search(r"[\+\-\*/]", e): continue
        v = calc(e)
        try:
            if abs(float(v) - float(mm[-1].group(2).replace(",", ""))) > 1e-6: return None
        except Exception: return None
        steps.append((l[:mm[-1].start()].strip().rstrip(":").strip() or "Next step:", e, v))
    if not steps or len(steps) > 3: return None
    try:
        if abs(float(steps[-1][2]) - float(ans.replace(",", "").replace("$", ""))) > 1e-6: return None
    except Exception: return None
    out = system(r) + f"<|user|>{q}<|end|>\n<|assistant|>"
    for i, (txt, e, v) in enumerate(steps):
        call = json.dumps({"name": "calculator", "arguments": {"expression": e}})
        out += f"<think>{txt}</think><tool_call>{call}</tool_call><tool_result>{v}</tool_result>"
    return out + f"The answer is {steps[-1][2]}.<|end|>"
if __name__ == "__main__":
    r = random.Random(20261124); n_s = n_g = 0
    p = r"D:\seed_v4\run4_data\gsm8k.txt"; size = os.path.getsize(p)
    raw = open(p, "rb").read(int(size * 0.99)).decode("utf-8", "ignore").split("<|endoftext|>")[:-1]
    with open(r"D:\seed_v4\seed_math1\data\synth_math.txt", "w", encoding="utf-8", newline="\n") as f:
        while n_s < 150000:
            d = synth(r)
            if d: f.write(d + "<|endoftext|>"); n_s += 1
    with open(r"D:\seed_v4\seed_math1\data\gsm8k_calc.txt", "w", encoding="utf-8", newline="\n") as f:
        for d in raw:
            o = from_gsm(d, r)
            if o: f.write(o + "<|endoftext|>"); n_g += 1
    print("synth", n_s, "gsm8k converted", n_g, "of", len(raw))
