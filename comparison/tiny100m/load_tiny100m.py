"""Load the tiny100m comparison model from the GitHub Release fp16 safetensors (or a local best.pt).

Example (from the seed repo root, after downloading the release weights):

    python comparison/tiny100m/load_tiny100m.py --weights tiny100m-best-step1350-fp16.safetensors --prompt "Once upon a time"

This model is comparison-only. It is not a Seed and does not use seed-socket-v1.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model100m import Tiny100M

DEFAULTS = dict(d=768, n_layer=10, n_head=12, mlp_hidden=1792, taps=(6, 13))


def _first_existing(*paths):
    for p in paths:
        if p is not None and Path(p).is_file():
            return Path(p)
    return None


def load_tiny100m(weights, config=None, tokenizer=None, device="cpu"):
    weights = Path(weights)
    cfg_path = _first_existing(
        Path(config) if config else None,
        weights.with_name("tiny100m-config.json"),
        HERE / "tiny100m-config.json",
        weights.parent / "tiny100m-config.json",
        weights.with_suffix(".json"),
    )
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path else {}

    if weights.suffix == ".pt":
        s = torch.load(weights, map_location="cpu", weights_only=False)
        sd = s.get("ema") or s["model"]
        vocab, block, core_cfg = s["vocab"], s["block"], s["core_cfg"]
        tower = dict(DEFAULTS)
    else:
        from safetensors.torch import load_file
        sd = load_file(str(weights))
        sd = {k: (v.float() if v.is_floating_point() else v) for k, v in sd.items()}
        if not cfg:
            raise SystemExit("tiny100m-config.json not found; pass --config")
        vocab, block, core_cfg = cfg["vocab"], cfg["block"], cfg["core_cfg"]
        tower = dict(DEFAULTS)
        tower.update(cfg.get("tower") or {})

    taps = tuple(tower.get("taps", DEFAULTS["taps"]))
    m = Tiny100M(
        core_cfg, vocab,
        d=int(tower.get("d", DEFAULTS["d"])),
        n_layer=int(tower.get("n_layer", DEFAULTS["n_layer"])),
        n_head=int(tower.get("n_head", DEFAULTS["n_head"])),
        mlp_hidden=int(tower.get("mlp_hidden", DEFAULTS["mlp_hidden"])),
        block=block, taps=taps,
    )
    m.load_state_dict(sd)
    m.eval()

    tok_path = _first_existing(
        Path(tokenizer) if tokenizer else None,
        weights.with_name("tiny100m-tokenizer.json"),
        HERE / "tiny100m-tokenizer.json",
        weights.parent / "tiny100m-tokenizer.json",
        weights.parent / "tokenizer" / "tokenizer.json",
        weights.parent / "tokenizer.json",
    )
    tok = Tokenizer.from_file(str(tok_path)) if tok_path else None
    return m.to(device), tok, cfg


def main():
    p = argparse.ArgumentParser(description="Load the tiny100m comparison model")
    p.add_argument("--weights", required=True, help="fp16 safetensors from the release, or a local best.pt")
    p.add_argument("--config", default=None)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--prompt", default=None)
    p.add_argument("--n", type=int, default=80)
    p.add_argument("--temp", type=float, default=0.7)
    a = p.parse_args()
    m, tok, cfg = load_tiny100m(a.weights, a.config, a.tokenizer, a.device)
    n = sum(p.numel() for p in m.parameters())
    print(f"loaded tiny100m  params={n:,}  step={cfg.get('step')}  device={a.device}")
    if a.prompt:
        if tok is None:
            raise SystemExit("tokenizer not found; pass --tokenizer")
        ids = torch.tensor([tok.encode(a.prompt).ids], device=a.device)
        out = m.generate(ids, a.n, temperature=a.temp)
        print(tok.decode(out[0].tolist(), skip_special_tokens=False))


if __name__ == "__main__":
    main()
