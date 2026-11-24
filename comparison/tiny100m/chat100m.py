"""Tool helpers (and an optional chat entry) for the tiny100m comparison model and for seed_pool.py --tools.

Wikipedia search is optional: set SEED_WIKI to a folder that has wiki_memory.py and a local leads index.
Calculator and read_file work with no extra data. Default workspace is ./workspace, or SEED_WORKSPACE.
"""
from __future__ import annotations
import argparse, ast, json, operator, os, re, sys
from pathlib import Path
import torch
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SPECS = {
    "wiki_search": {"name": "wiki_search", "description": "Search Wikipedia article introductions and return the best matches", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "what to look up"}}, "required": ["query"]}},
    "calculator": {"name": "calculator", "description": "Evaluate an arithmetic expression exactly", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "e.g. 12*(3+4)"}}, "required": ["expression"]}},
    "read_file": {"name": "read_file", "description": "Read a text file from the user's workspace", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "file name"}}, "required": ["path"]}},
}
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
       ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv, ast.USub: operator.neg, ast.UAdd: operator.pos}


def calculator(expression):
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in OPS:
            if isinstance(n.op, ast.Pow) and abs(ev(n.right)) > 100:
                raise ValueError("exponent too large")
            return OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in OPS:
            return OPS[type(n.op)](ev(n.operand))
        raise ValueError("unsupported expression")
    try:
        r = ev(ast.parse(str(expression).replace("^", "**").replace("x", "*"), mode="eval").body)
        return str(int(r)) if float(r).is_integer() else f"{r:.6g}"
    except Exception as e:
        return f"Error: {e}"


_wm = None
def wiki_search(query):
    global _wm
    wiki = os.environ.get("SEED_WIKI")
    if not wiki:
        return "Error: Wikipedia search is not configured (set SEED_WIKI to a local leads index)."
    if _wm is None:
        sys.path.insert(0, wiki)
        import wiki_memory
        _wm = wiki_memory.default_memory()
    res = _wm.search(str(query), 3)
    if not res:
        return "No results."
    out = []
    for i, (t, x, _) in enumerate(res):
        x = x if len(x) <= 320 else x[:320].rsplit(" ", 1)[0] + " ..."
        out.append(f"{i+1}. {t}: {x}")
    return "\n".join(out)


def read_file(path, workspace):
    p = (Path(workspace) / str(path)).resolve()
    root = Path(workspace).resolve()
    if root not in p.parents and p != root:
        return "Error: access outside the workspace is not allowed"
    if not p.is_file():
        return f"Error: file not found: {path}"
    return p.read_text(encoding="utf-8", errors="ignore")[:1500]


def run_tool(call_json, workspace):
    try:
        c = json.loads(call_json)
        name, args = c.get("name"), c.get("arguments", {})
        if isinstance(args, str):
            args = json.loads(args)
    except Exception as e:
        return f"Error: could not parse tool call ({e})"
    if name == "wiki_search":
        return wiki_search(args.get("query", ""))
    if name == "calculator":
        return calculator(args.get("expression", ""))
    if name == "read_file":
        return read_file(args.get("path", ""), workspace)
    return f"Error: unknown tool {name}"


def main():
    from load_tiny100m import load_tiny100m
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--workspace", default=os.environ.get("SEED_WORKSPACE", str(Path.cwd() / "workspace")))
    p.add_argument("--once", default=None)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    Path(a.workspace).mkdir(parents=True, exist_ok=True)
    m, tok, _ = load_tiny100m(a.weights, a.config, a.tokenizer, a.device)
    if tok is None:
        raise SystemExit("tokenizer not found")
    system = "<|system|>You are a helpful assistant. You can call these tools:\n" + "\n".join(json.dumps(v) for v in SPECS.values()) + "<|end|>\n"
    hist = system
    qs = [a.once] if a.once else None
    end, tc_end = tok.token_to_id("<|end|>"), tok.token_to_id("</tool_call>")
    while True:
        q = qs.pop(0) if qs else (None if a.once else input("you> "))
        if not q or q.strip().lower() in ("exit", "quit"):
            break
        hist += f"<|user|>{q}<|end|>\n"
        text = hist + "<|assistant|>"
        reply = ""
        for _ in range(5):
            ids = torch.tensor([tok.encode(text).ids], device=a.device)
            out = m.generate(ids, 200, temperature=a.temperature, stop_ids=(end, tc_end) if end is not None else ())
            new = tok.decode(out[0, ids.size(1):].tolist(), skip_special_tokens=False)
            text += new
            reply += new
            mt = re.search(r"<tool_call>(.*?)</tool_call>$", new, re.S)
            if not mt:
                break
            result = run_tool(mt.group(1), a.workspace)
            print(f"  [tool] {mt.group(1).strip()}\n  [result] {result[:300]}")
            piece = f"<tool_result>{result}</tool_result>"
            text += piece
            reply += piece
        if not text.endswith("<|end|>"):
            text += "<|end|>"
        hist = text + "\n"
        final = re.sub(r"<think>.*?</think>|<tool_call>.*?</tool_call>|<tool_result>.*?</tool_result>", "", reply, flags=re.S).replace("<|end|>", "").strip()
        print("bot>", final)
        if a.once:
            break


if __name__ == "__main__":
    main()
