#!/usr/bin/env python3
"""Export a Seed v2.0 checkpoint as a shareable fp16 organ (weights only).

Drops optimizer state and RNG. The published .safetensors + .json pair is
typically well under GitHub's 100 MB limit (this script writes fp16).

Seed v2.0 organs are a different line from Pool 5 / Seed v4. They are not
loaded by `chat.py` or `seed_pool.py`. Use `v2/chat_v2.py`.

Example (from the repo root):

    python scripts/export_seed_v2.py runs/reason1/best_ce.pt organs/seeds/myseed \
        --name myseed --description "what this seed learned" \
        --tokenizer-dir tokenizers/seed-bpe8192-r1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "v2"))
import seed_v2 as S  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
import torch  # noqa: E402


def sha256_file(path, n=64):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:n]


def main():
    ap = argparse.ArgumentParser(description="Export a Seed v2.0 checkpoint as an fp16 organ")
    ap.add_argument("checkpoint", help="best_ce.pt / latest.pt (or any v2 checkpoint)")
    ap.add_argument("dest", help="output stem, e.g. organs/seeds/myseed (no suffix)")
    ap.add_argument("--name", default=None, help="organ name (default: dest filename)")
    ap.add_argument("--description", default="Whole frozen Seed v2.0 seed, EMA weights, fp16.")
    ap.add_argument("--tokenizer-dir", default=None, help="repo-relative tokenizer directory to record in the manifest")
    ap.add_argument("--no-ema", action="store_true", help="export raw weights instead of EMA")
    a = ap.parse_args()

    src = Path(a.checkpoint)
    dest = Path(a.dest)
    if dest.suffix in {".json", ".safetensors", ".pt"}:
        dest = dest.with_suffix("")
    dest.parent.mkdir(parents=True, exist_ok=True)
    name = a.name or dest.name

    ck = torch.load(src, map_location="cpu", weights_only=False)
    model = S.SeedV2.from_structure(ck["structure"])
    model.load_state_dict(ck["model"])
    used_ema = False
    if ck.get("ema") and not a.no_ema:
        ema = S.EMA(model, 0.999)
        ema.load_by_name(model, ck["ema"])
        ema.swap(model)
        used_ema = True
    step = ck.get("step")
    if step is None and isinstance(ck.get("S"), dict):
        step = ck["S"].get("step")

    st = model.structure()
    tok = dict(st.get("tokenizer") or {})
    if a.tokenizer_dir:
        tok["dir"] = a.tokenizer_dir
        tjson = REPO / a.tokenizer_dir / "tokenizer.json"
        if tjson.exists():
            tok["sha256_full"] = sha256_file(tjson, 64)
            tok["sha256"] = tok["sha256_full"][:16]
    st["tokenizer"] = tok

    tensors = {}
    n_params = 0
    for k, v in model.state_dict().items():
        n_params += int(v.numel())
        tensors[k] = (
            v.detach().half().cpu().contiguous()
            if v.is_floating_point()
            else v.detach().cpu().contiguous()
        )
    st_p = Path(str(dest) + ".safetensors")
    js_p = Path(str(dest) + ".json")
    save_file(tensors, str(st_p), metadata={"format": "seed-v2-part/1", "kind": "tree", "dtype": "fp16", "arch": "seed-v2.0"})
    man = dict(
        kind="tree",
        id=f"tree-{model.lineage}",
        name=name,
        description=a.description,
        structure=st,
        tokenizer=tok,
        lineage=model.lineage,
        format="seed-v2-part/1",
        arch="seed-v2.0",
        file=st_p.name,
        sha256=sha256_file(st_p, 16),
        sha256_full=sha256_file(st_p, 64),
        n_params=n_params,
        dtype="fp16",
        ema=used_ema,
        step=step,
        metrics=ck.get("metrics"),
        created=time.strftime("%Y-%m-%d %H:%M:%S"),
        license="Apache-2.0",
        socket=None,
        shape=f"{st['L']}x{st['K']}",
    )
    js_p.write_text(json.dumps(man, indent=1) + "\n", encoding="utf-8")
    mb = st_p.stat().st_size / (1 << 20)
    print(f"wrote {st_p}  {mb:.2f} MB  params={n_params}  step={man['step']}  sha={man['sha256_full'][:16]}")
    print(f"wrote {js_p}")


if __name__ == "__main__":
    main()
