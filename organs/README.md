# Starter organs

Colby's starter organs are free.

An **organ** is a whole frozen seed used in a Seed Pool. An **organelle** is the older per-column unit (one cell per layer plus a rank-8 adapter), shipped as a `seed-organ/1.1` bundle. Filenames are unchanged so existing loaders still work.

| path | what |
|---|---|
| `seeds/run3-trunk-final.json` (+ `.safetensors`) | run3 fluent trunk, starter **organ**, fp16 Seed v4 tree part (4×2, step 74,300) |
| `seeds/run4-tools-reasoning-final.json` (+ `.safetensors`) | run4 tools + reasoning, starter **organ**, fp16 (6×1, step 19,822) |
| `run4-tools-reasoning-1.0.1.json` (+ `.safetensors`) | run4 **organelle bundle** (`seed-organ/1.1`) with a rehearsal cache |

Load an organ:

```bash
python chat.py organs/seeds/run3-trunk-final.json
```

Add one to a pool (see `pool1/registry.json` and `seed_pool.py`):

```bash
python seed_pool.py add myseed organs/seeds/run4-tools-reasoning-final.json
```

`parts_tool.py import` can transplant an organelle bundle into a compatible socket trunk. This path is **experimental** and is **known to fail for whole-seed merges** (merged1: 0/20 correct queries, 1/20 answers, 1/10 math). Prefer the Seed Pool to combine organs. Details: [pool1/results/SUMMARY.md](../pool1/results/SUMMARY.md#failed-merge-merged1).

```bash
python parts_tool.py import organs/run4-tools-reasoning-1.0.1 --host path/to/host.pt --out runs/merged.pt
```

The weights are fp16 safetensors. `chat.py`, `seed_pool.py`, and `parts_tool.py` cast them back to fp32 on load. See [ORGANS_README.md](../ORGANS_README.md) and [SOCKET_SPEC.md](../SOCKET_SPEC.md) for the formats.
