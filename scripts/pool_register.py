#!/usr/bin/env python3
"""Register / list / remove frozen seeds in a pool registry.json.

Seeds must share the official tokenizer and socket if you want them in the
published Pool 5 (seed-bpe4096-v1 + seed-socket-v1). After adding or removing
a seed, retrain the looping Top-6 router (`pool5/eval/pool5_loop.py base` then
`loop`) because the router keys are per-name.

Examples (from the repo root):

    python scripts/pool_register.py --dir pool5 list
    python scripts/pool_register.py --dir pool5 add myseed organs/seeds/myseed.json --desc "what it learned"
    python scripts/pool_register.py --dir pool5 remove myseed
    python scripts/pool_register.py --dir pool5 init   # write a registry from the 16 published Pool 5 organs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import seed_v4 as S  # noqa: E402

POOL5_DEFAULTS = [
    ("run3", "organs/seeds/run3-trunk-final.json", "fluent TinyStories trunk (4x2, o1+o2)"),
    ("run4", "organs/seeds/run4-tools-reasoning-final.json", "tools + reasoning seed (6x1): wiki_search/calculator, gsm8k steps"),
    ("json1", "organs/seeds/json1-final.json", "JSON/tool-call specialist: argument filling, read/write/fix JSON"),
    ("answer1", "organs/seeds/answer1-final.json", "answer-from-results: short final answer from <tool_result>"),
    ("math1", "organs/seeds/math1-final.json", "math word problems -> plan + calculator calls"),
    ("reason1", "organs/seeds/reason1-final.json", "step-by-step reasoning puzzles + gsm8k chains"),
    ("plan1", "organs/seeds/plan1-final.json", "word-problem planner (4.43M)"),
    ("py1", "organs/seeds/py1-final.json", "Python writer ~10M"),
    ("plan2", "organs/seeds/plan2-final.json", "word-problem planner ~10M"),
    ("fix1", "organs/seeds/fix1-final.json", "run-and-fix with python_exec"),
    ("inst1", "organs/seeds/inst1-final.json", "instruction follower"),
    ("sh1", "organs/seeds/sh1-final.json", "Bash/shell writer (retrained, UTF-8 fix)"),
    ("c1", "organs/seeds/c1-final.json", "C writer (retrained)"),
    ("js1", "organs/seeds/js1-final.json", "JavaScript writer (retrained)"),
    ("ps1", "organs/seeds/ps1-final.json", "PowerShell writer (retrained, held-out early stop)"),
    ("alg1", "organs/seeds/alg1-final.json", "language-agnostic algorithmic reasoning (real APPS mix, 6x2)"),
]


def rel_to_repo(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def load_reg(d: Path) -> dict:
    p = d / "registry.json"
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"seeds": {}, "budget_mb": 2048.0}


def save_reg(d: Path, reg: dict):
    d.mkdir(parents=True, exist_ok=True)
    (d / "registry.json").write_text(json.dumps(reg, indent=1) + "\n", encoding="utf-8")
    print("wrote", d / "registry.json")


def describe(path: Path, name: str, desc: str) -> dict:
    m, info = S.load_model_any(str(path), use_ema=True, device="cpu")
    params = int(sum(p.numel() for p in m.parameters()))
    mb = round(params * 4 / 2**20, 1)
    tok = (m.tokenizer or {}).get("sha256")
    anc = (m.anchor or {}).get("sha256") if getattr(m, "anchor", None) else None
    rec = dict(
        name=name,
        description=desc,
        path=rel_to_repo(path) if path.suffix == ".json" else rel_to_repo(path),
        params=params,
        weights_mb=mb,
        shape=m.shape_str() if hasattr(m, "shape_str") else None,
        ctx=getattr(m, "ctx", None),
        step=(info or {}).get("step") if isinstance(info, dict) else None,
        active=True,
        tokenizer=tok,
        anchor=anc,
    )
    del m
    return rec


def check_compat(reg: dict):
    toks = {s.get("tokenizer") for s in reg["seeds"].values() if s.get("tokenizer")}
    anch = {s.get("anchor") for s in reg["seeds"].values() if s.get("anchor")}
    if len(toks) > 1:
        print("WARNING: tokenizer shas are not unique:", toks)
    if len(anch) > 1:
        print("WARNING: anchor shas are not unique:", anch)
    tot = sum(s.get("weights_mb") or 0 for s in reg["seeds"].values() if s.get("active", True))
    print(f"{len(reg['seeds'])} seeds, active weights {tot:.1f} MB of budget {reg.get('budget_mb', 0):.0f} MB")


def add(d: Path, name: str, path: str, desc: str):
    src = Path(path)
    if not src.is_absolute():
        cand = (Path.cwd() / src)
        src = cand if cand.exists() else (REPO / src)
    if not src.is_file():
        raise SystemExit(f"seed not found: {path}")
    reg = load_reg(d)
    rec = describe(src, name, desc or "")
    # prefer the .json manifest next to a .safetensors / if user passed a .pt keep it
    if src.suffix == ".safetensors" and src.with_suffix(".json").is_file():
        rec["path"] = rel_to_repo(src.with_suffix(".json"))
    reg.setdefault("seeds", {})[name] = rec
    save_reg(d, reg)
    check_compat(reg)
    print("NOTE: retrain the looping router after changing the seed set:")
    print(f"  python pool5/eval/pool5_loop.py base --dir {d} --mode single --init-k 6 --step-k 3")
    print(f"  python pool5/eval/pool5_loop.py loop --dir {d} --minutes 15 --mix-json path/to/mix.json")


def main():
    ap = argparse.ArgumentParser(description="Add, list, or remove seeds in a pool registry.json")
    ap.add_argument("cmd", choices=["list", "add", "remove", "init"])
    ap.add_argument("name", nargs="?")
    ap.add_argument("path", nargs="?")
    ap.add_argument("--dir", default="pool5")
    ap.add_argument("--desc", default="")
    ap.add_argument("--budget-mb", type=float, default=2048.0)
    a = ap.parse_args()
    d = Path(a.dir)
    if not d.is_absolute():
        d = (Path.cwd() / d).resolve()
    if a.cmd == "init":
        reg = {"seeds": {}, "budget_mb": a.budget_mb}
        for name, path, desc in POOL5_DEFAULTS:
            src = REPO / path
            if not src.is_file():
                print("skip missing", path); continue
            reg["seeds"][name] = describe(src, name, desc)
        save_reg(d, reg); check_compat(reg); return
    if a.cmd == "list":
        reg = load_reg(d)
        if not reg["seeds"]:
            print("empty registry at", d / "registry.json"); return
        for n, s in reg["seeds"].items():
            flag = "ACTIVE" if s.get("active", True) else "idle"
            print(f"{n:10s} {flag:6s} {s.get('shape','?'):6s} {s.get('weights_mb',0):7.1f} MB  {s.get('path')}  {s.get('description','')}")
        check_compat(reg); return
    if a.cmd == "add":
        if not a.name or not a.path:
            raise SystemExit("usage: pool_register.py add NAME PATH [--desc ...]")
        add(d, a.name, a.path, a.desc); return
    if a.cmd == "remove":
        if not a.name:
            raise SystemExit("usage: pool_register.py remove NAME")
        reg = load_reg(d)
        if a.name not in reg["seeds"]:
            raise SystemExit(f"no seed named {a.name}")
        del reg["seeds"][a.name]
        save_reg(d, reg); check_compat(reg)
        print("NOTE: retrain the looping router after changing the seed set.")


if __name__ == "__main__":
    main()
