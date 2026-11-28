#!/usr/bin/env python3
"""Chat with a Seed v2.0 organ (.json + .safetensors) or a raw v2 checkpoint.

Not a substitute for chat.py (that loads Seed v4 / Pool organs).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "v2"))
import seed_v2 as S  # noqa: E402


def load_organ(path, device="cpu"):
    p = Path(path)
    if p.suffix == ".json":
        man = json.loads(p.read_text(encoding="utf-8"))
        weights = p.with_name(man.get("file") or p.with_suffix(".safetensors").name)
        st = man["structure"]
        model = S.SeedV2.from_structure(st)
        sd = load_file(str(weights))
        model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
        tok_dir = (man.get("tokenizer") or {}).get("dir") or "tokenizers/seed-bpe8192-r1"
        tok_path = REPO / tok_dir / "tokenizer.json"
        return model.to(device).eval(), Tokenizer.from_file(str(tok_path)), man
    ck = torch.load(p, map_location="cpu", weights_only=False)
    model = S.SeedV2.from_structure(ck["structure"])
    model.load_state_dict(ck["model"])
    if ck.get("ema"):
        ema = S.EMA(model, 0.999)
        ema.load_by_name(model, ck["ema"])
        ema.swap(model)
    tok_info = ck.get("tokenizer") or ck["structure"].get("tokenizer") or {}
    tok_path = tok_info.get("path") or str(REPO / "tokenizers/seed-bpe8192-r1/tokenizer.json")
    return model.to(device).eval(), Tokenizer.from_file(tok_path), ck


def main():
    ap = argparse.ArgumentParser(description="Chat with a Seed v2.0 organ")
    ap.add_argument("organ", help="organs/seeds/foo.json or a v2 .pt checkpoint")
    ap.add_argument("--prompt", default="User: Janet has 12 apples and gives away 5. Then she buys 9 more. How many apples does she have now?\nAssistant:")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    a.prompt = a.prompt.replace("\\n", "\n")
    model, tok, _ = load_organ(a.organ, a.device)
    S.DEV = torch.device(a.device)
    eos = tok.token_to_id("<eos>")
    ids = torch.tensor([tok.encode(a.prompt, add_special_tokens=False).ids], device=torch.device(a.device))
    out = model.generate(ids, a.n, temp=a.temp, stop_id=eos)
    print(tok.decode(out[0].tolist(), skip_special_tokens=False))


if __name__ == "__main__":
    main()
