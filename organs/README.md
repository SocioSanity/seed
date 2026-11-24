# Starter organs

Colby's starter organs are free.

| path | what |
|---|---|
| `seeds/run3-trunk-final.json` (+ `.safetensors`) | run3 fluent trunk, whole frozen seed, fp16, Seed v4 tree part (4×2, step 74,300) |
| `seeds/run4-tools-reasoning-final.json` (+ `.safetensors`) | run4 tools + reasoning, whole frozen seed, fp16 (6×1, step 19,822) |
| `run4-tools-reasoning-1.0.1.json` (+ `.safetensors`) | run4's organ column (`seed-organ/1.1`) with a rehearsal cache |

Load a whole seed:

```bash
python chat.py organs/seeds/run3-trunk-final.json
```

Add one to a pool (see `pool1/registry.json` and `seed_pool.py`):

```bash
python seed_pool.py add myseed organs/seeds/run4-tools-reasoning-final.json
```

Transplant the organ bundle into a compatible socket trunk:

```bash
python parts_tool.py import organs/run4-tools-reasoning-1.0.1 --host path/to/host.pt --out runs/merged.pt
```

The weights are fp16 safetensors. `chat.py`, `seed_pool.py`, and `parts_tool.py` cast them back to fp32 on load. See [ORGANS_README.md](../ORGANS_README.md) and [SOCKET_SPEC.md](../SOCKET_SPEC.md) for the formats.
