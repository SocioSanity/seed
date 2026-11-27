#!/usr/bin/env python3
"""Export a trained Seed checkpoint as a shareable fp16 organ (tree part).

Drops optimizer state. The published .safetensors + .json pair is typically
8-20 MB and is what `seed_pool.py add` and `chat.py` expect.

Example (from the repo root):

    python scripts/export_seed.py runs/myseed/latest.pt organs/seeds/myseed \\
        --name myseed --description "what this seed learned"
    python seed_pool.py --dir pool5 add myseed organs/seeds/myseed.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import seed_v4 as S  # noqa: E402
from safetensors.torch import save_file  # noqa: E402


def sha256_file(path, n=64):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:n]


def main():
    ap = argparse.ArgumentParser(description="Export a Seed checkpoint as an fp16 organ")
    ap.add_argument("checkpoint", help="latest.pt or any Seed checkpoint / tree part")
    ap.add_argument("dest", help="output stem, e.g. organs/seeds/myseed (no suffix)")
    ap.add_argument("--name", default=None, help="organ name (default: dest filename)")
    ap.add_argument("--description", default="Whole frozen seed, EMA weights, fp16.")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    src = Path(a.checkpoint)
    dest = Path(a.dest)
    if dest.suffix in {".json", ".safetensors", ".pt"}:
        dest = dest.with_suffix("")
    dest.parent.mkdir(parents=True, exist_ok=True)
    name = a.name or dest.name

    m, info = S.load_model_any(str(src), use_ema=True, device=a.device)
    st = m.structure()
    tok = dict(st.get("tokenizer") or {})
    if not tok.get("dir"):
        tok["dir"] = "tokenizers/seed-bpe4096-v1"
    st["tokenizer"] = tok

    tensors = {}
    n_params = 0
    for k, v in m.state_dict().items():
        n_params += int(v.numel())
        tensors[k] = (
            v.detach().half().cpu().contiguous()
            if v.is_floating_point()
            else v.detach().cpu().contiguous()
        )
    st_p = Path(str(dest) + ".safetensors")
    js_p = Path(str(dest) + ".json")
    save_file(tensors, str(st_p), metadata={"format": "seed-part/1", "kind": "tree", "dtype": "fp16"})
    man = dict(
        kind="tree",
        id=f"tree-{m.lineage}",
        name=name,
        description=a.description,
        structure=st,
        sig=S.model_sig(m),
        tokenizer=tok,
        lineage=m.lineage,
        format="seed-part/1",
        file=st_p.name,
        sha256=S.file_sha(st_p),
        sha256_full=sha256_file(st_p, 64),
        n_params=n_params,
        dtype="fp16",
        ema=True,
        step=info.get("step") if isinstance(info, dict) else None,
        created=time.strftime("%Y-%m-%d %H:%M:%S"),
        license="Apache-2.0",
        socket=st.get("socket"),
        shape=m.shape_str() if hasattr(m, "shape_str") else None,
    )
    js_p.write_text(json.dumps(man, indent=1), encoding="utf-8")
    mb = st_p.stat().st_size / (1 << 20)
    print(f"wrote {st_p}  {mb:.2f} MB  params={n_params}  step={man['step']}  sha={man['sha256_full'][:16]}")
    print(f"wrote {js_p}")


if __name__ == "__main__":
    main()
