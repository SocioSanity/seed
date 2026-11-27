"""Tool runtimes for Seed Pool chats and for specialist training recipes.

Wikipedia search is optional: set SEED_WIKI to a folder that has wiki_memory.py
and a local leads index. Calculator, read_file, and python_exec work with no
extra data. Default workspace is ./workspace, or SEED_WORKSPACE.
"""
from __future__ import annotations

import ast
import json
import operator
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

SPECS = {
    "wiki_search": {
        "name": "wiki_search",
        "description": "Search Wikipedia article introductions and return the best matches",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "what to look up"}},
            "required": ["query"],
        },
    },
    "calculator": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression exactly",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "e.g. 12*(3+4)"}},
            "required": ["expression"],
        },
    },
    "read_file": {
        "name": "read_file",
        "description": "Read a text file from the user's workspace",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "file name"}},
            "required": ["path"],
        },
    },
    "python_exec": {
        "name": "python_exec",
        "description": "Run Python code in a sandbox and return stdout/stderr (tracebacks included).",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
    },
}

OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


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


def python_exec(code, timeout=10):
    """Run a snippet in an isolated interpreter (python -I, temp dir, timeout)."""
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "main.py")
        with open(p, "w", encoding="utf-8") as f:
            f.write(str(code))
        try:
            r = subprocess.run(
                [sys.executable, "-I", p],
                cwd=td,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            out = (r.stdout + r.stderr).replace(td, "/sandbox").replace(p, "/sandbox/main.py")
            out = re.sub(r'File "[^"]*main\.py"', 'File "main.py"', out)
            body = out.strip()[-700:]
            if r.returncode != 0 and not body:
                return f"Error: process exited with code {r.returncode}"
            return body or "(no output)"
        except subprocess.TimeoutExpired:
            return "TimeoutError: execution exceeded 10 seconds"


def run_tool(call_json, workspace=None):
    if workspace is None:
        workspace = os.environ.get("SEED_WORKSPACE", str(REPO / "workspace"))
    Path(workspace).mkdir(parents=True, exist_ok=True)
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
    if name == "python_exec":
        return python_exec(args.get("code", ""))
    return f"Error: unknown tool {name}"
