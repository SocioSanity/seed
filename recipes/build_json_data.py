"""Synthetic JSON / tool-call data for seed_json1 (2026-11-24). Writing and reading JSON: tool calls with argument filling,
JSON tool results read back into answers, JSON reading questions, and 'write this as JSON' tasks (objects, arrays, nesting)."""
import json, random
R = random.Random(11)
SPECS = {"calculator": {"name": "calculator", "description": "Evaluate an arithmetic expression exactly", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "e.g. 12*(3+4)"}}, "required": ["expression"]}},
         "wiki_search": {"name": "wiki_search", "description": "Search Wikipedia article introductions and return the best matches", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "what to look up"}}, "required": ["query"]}},
         "read_file": {"name": "read_file", "description": "Read a text file from the user's workspace", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "file name"}}, "required": ["path"]}}}
VERBS = ["get", "search", "find", "create", "update", "convert", "check", "calculate", "book", "send", "list", "translate", "schedule", "track"]
NOUNS = ["weather", "stock_price", "recipe", "flight", "hotel", "movie", "book", "user", "order", "event", "song", "currency", "news", "package", "invoice", "reminder"]
WORDS = "red blue green apple river stone cloud tiger maple ocean violet silver paper window garden rocket candle forest piano yellow castle bridge lemon".split()
CITIES = ["Paris", "Tokyo", "Chicago", "Lagos", "Lima", "Oslo", "Austin", "Cairo", "Seoul", "Denver", "Madrid", "Toronto"]
NAMES = ["Alice", "Bob", "Maria", "Ken", "Priya", "Omar", "Lena", "Sam", "Chen", "Ivy", "Noah", "Zara"]
def rval(t):
    return {"string": lambda: R.choice(WORDS + CITIES + NAMES), "number": lambda: R.choice([R.randint(0, 999), round(R.uniform(0, 500), 2)]),
            "integer": lambda: R.randint(1, 5000), "boolean": lambda: R.choice([True, False]),
            "array": lambda: [R.choice(WORDS) for _ in range(R.randint(1, 4))]}[t]()
def robj(depth=0):
    o = {}
    for _ in range(R.randint(2, 5)):
        k = R.choice(WORDS + ["id", "name", "count", "price", "status", "items", "tags", "city", "date", "total", "score"])
        r = R.random()
        if depth < 2 and r < 0.18: o[k] = robj(depth + 1)
        elif depth < 2 and r < 0.32: o[k] = [robj(depth + 1) if R.random() < 0.4 else rval("string") for _ in range(R.randint(1, 3))]
        else: o[k] = rval(R.choice(["string", "number", "integer", "boolean", "array"]))
    return o
def dumps(o): return json.dumps(o, ensure_ascii=False)
def sysp(specs): return "<|system|>You are a helpful assistant. You can call these tools:\n" + "\n".join(dumps(s) for s in specs) + "<|end|>\n"
def tool_spec():
    name = f"{R.choice(VERBS)}_{R.choice(NOUNS)}"; props = {}; req = []
    for _ in range(R.randint(1, 3)):
        pn = R.choice(["city", "name", "query", "amount", "date", "limit", "symbol", "title", "user_id", "tags", "include_details", "language", "from", "to"])
        t = {"amount": "number", "limit": "integer", "user_id": "integer", "tags": "array", "include_details": "boolean"}.get(pn, "string")
        props[pn] = {"type": t, "description": f"the {pn.replace('_', ' ')}"}
        if t == "array": props[pn]["items"] = {"type": "string"}
        req.append(pn)
    return {"name": name, "description": f"{name.replace('_', ' ').capitalize()}", "parameters": {"type": "object", "properties": props, "required": sorted(set(req))}}
def ex_toolcall():
    sp = tool_spec(); args = {p: rval(v["type"]) for p, v in sp["parameters"]["properties"].items()}
    parts = ", ".join(f"{p.replace('_', ' ')} {', '.join(v) if isinstance(v, list) else str(v).lower() if isinstance(v, bool) else v}" for p, v in args.items())
    q = R.choice([f"Please {sp['name'].split('_')[0]} the {sp['name'].split('_', 1)[1].replace('_', ' ')} with {parts}.", f"Can you {sp['name'].replace('_', ' ')} for {parts}?", f"I need to {sp['name'].replace('_', ' ')}: {parts}."])
    res = robj(); k = R.choice(list(res)); v = res[k]
    ans = f"The {k} is {dumps(v) if isinstance(v, (dict, list)) else v}."
    extra = [SPECS[R.choice(list(SPECS))]] if R.random() < 0.5 else []
    return (sysp([sp] + extra) + f"<|user|>{q}<|end|>\n<|assistant|><tool_call>{dumps({'name': sp['name'], 'arguments': args})}</tool_call>"
            f"<tool_result>{dumps(res)}</tool_result>{ans}<|end|>")
def ex_calc():
    a, b, c = R.randint(2, 999), R.randint(2, 99), R.randint(1, 50); op = R.choice(["+", "-", "*"]); op2 = R.choice(["+", "-", "*"])
    e = R.choice([f"{a}{op}{b}", f"{a}{op}{b}{op2}{c}", f"({a}{op}{b}){op2}{c}"]); v = eval(e)
    q = R.choice([f"Compute {e.replace('*', ' * ').replace('+', ' + ').replace('-', ' - ')}", f"What is {e}?", f"Calculate {e}."])
    return (sysp([SPECS["calculator"], SPECS["wiki_search"]]) + f"<|user|>{q}<|end|>\n<|assistant|><think>This needs exact arithmetic, so I will use the calculator.</think>"
            f"<tool_call>{dumps({'name': 'calculator', 'arguments': {'expression': e}})}</tool_call><tool_result>{v}</tool_result>{e} = {v}.<|end|>")
def ex_read():
    o = robj(); k = R.choice(list(o)); v = o[k]
    q = R.choice([f"What is the value of \"{k}\" in this JSON?\n{dumps(o)}", f"Here is some JSON: {dumps(o)}\nWhat does \"{k}\" contain?"])
    return f"<|user|>{q}<|end|>\n<|assistant|>\"{k}\" is {dumps(v)}.<|end|>"
def ex_write():
    o = robj(); desc = "; ".join(f"{k}: {', '.join(map(str, v)) if isinstance(v, list) and all(not isinstance(x, dict) for x in v) else dumps(v)}" for k, v in o.items())
    q = R.choice([f"Write this as a JSON object: {desc}", f"Convert to JSON: {desc}"])
    return f"<|user|>{q}<|end|>\n<|assistant|>{json.dumps(o, ensure_ascii=False, indent=R.choice([None, 2]))}<|end|>"
def ex_fix():
    o = robj(); s = dumps(o); bad = s[:-1] if R.random() < 0.5 else s.replace("{", "[", 1)
    return f"<|user|>Fix this JSON so it is valid: {bad}<|end|>\n<|assistant|>{s}<|end|>"
GEN = [(ex_toolcall, 0.42), (ex_calc, 0.2), (ex_read, 0.16), (ex_write, 0.14), (ex_fix, 0.08)]
fs, ws = zip(*GEN)
with open(r"D:\seed_v4\seed_json1\data\synth_json.txt", "w", encoding="utf-8", newline="\n") as f:
    for i in range(120000):
        f.write(R.choices(fs, ws)[0]() + "<|endoftext|>")
print("ok")
