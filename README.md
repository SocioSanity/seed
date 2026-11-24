# Seed

A tiny self-growing language model with plug-and-play **organs** on a standard socket (`seed-socket-v1`).

Train small specialist seeds for one skill or tool, then transplant their organs into a trunk. Organs ship with a rehearsal cache so the skill does not wash out after it lands. The goal is capable, customizable AI on small computers — not a giant general model.

## How it works

- The **trunk** is a small growing transformer (Seed v4). Width is a whole number of organs.
- An **organ** is one column: one cell in every layer, plus a rank-8 adapter, on a shared frozen embedding anchor.
- Specialists are trained on one skill or tool, then harvested with `parts_tool.py` into a signed `seed-organ` bundle.
- Bundles plug into any Seed that uses the same socket and tokenizer (`seed-bpe4096-v1`).
- A rehearsal cache of hard examples lives inside the bundle so later training does not erode the skill.

## Vision

Named organs, loaded and unloaded from disk by a small **manager** organ that can prefetch what it will need next. Only the organs a task needs sit in memory, so a small PC can run a capable, customized AI.

Colby's starter organs are free. Anyone can train, share, or sell their own organs. The engine and the socket spec are open source.

## Hardware

Built and trained on a desktop with an **NVIDIA GTX 1660 Ti (6 GB)**. Training and inference also run on CPU (slower).

## Current results

**run3** (fluent trunk, TinyStories):

- 4.43M parameters (shape 4×2)
- Best TinyStories EMA loss **1.5627** = **0.570 bits/byte**

## Quickstart

### Install

Python 3.10+ (developed on 3.12).

```bash
pip install -r requirements.txt
```

For GPU training, install a CUDA build of PyTorch from [pytorch.org](https://pytorch.org) (this project was trained with `torch 2.6.0+cu124`). The `requirements.txt` pin is CPU-agnostic; swap in the CUDA wheel if you have a GPU.

### Train

The standard tokenizer is included at `tokenizers/seed-bpe4096-v1`. Point the trainer at text with `--mix`, or at `train.bin` / `valid.bin` from `prep_data.py` with `--data`.

```bash
# GPU, one hour, a folder or text file of training data
python seed_v4.py --out runs/myrun --device cuda --minutes 60 --mix path/to/text.txt

# CPU
python seed_v4.py --out runs/myrun --device cpu --minutes 45 --threads 3 --mix path/to/text.txt
```

A PowerShell launcher with configurable paths lives at `scripts/run_seed_v4.ps1`.

Prepare TinyStories (or similar) into token files:

```bash
python prep_data.py --train-text path/to/TinyStoriesV2-GPT4-train.txt --valid-text path/to/TinyStoriesV2-GPT4-valid.txt --out data
python seed_v4.py --out runs/myrun --device cuda --minutes 60 --data data
```

Or train on your own folder of `.txt`, `.md`, or `.pdf` files:

```bash
python prep_data.py --data-dir path/to/your_texts --out data
```

Stop a run cleanly by creating the file `runs/myrun/STOP` (it saves, then exits). Resume with `--resume` and the same `--out`.

### Chat

```bash
python chat.py runs/myrun/latest.pt
python chat.py runs/myrun/latest.pt --prompt "Once upon a time" --n 120
python chat.py runs/myrun/latest.pt --parts parts/exports/organ_a.json --device cpu
```

### Harvest and transplant organs

```bash
# Publish a socket once from a trunk (needed for compatible organs)
python parts_tool.py make-socket --from runs/run1/latest.pt --socket-dir socket/seed-socket-v1

# Train a specialist organ on a task
python parts_tool.py train-organ --host runs/specialist/latest.pt --task task.jsonl --task-eval task_eval.jsonl --name wiki-search --out organs/my/wiki-search-1.0.0

# Or harvest an organ already inside a trained seed
python parts_tool.py harvest --host runs/specialist/latest.pt --organ o1 --out organs/my/wiki-search-1.0.0 --name wiki-search

# Transplant into a trunk
python parts_tool.py import organs/my/wiki-search-1.0.0 --host runs/trunk/latest.pt --out runs/merged.pt
```

See [ORGANS_README.md](ORGANS_README.md), [SOCKET_SPEC.md](SOCKET_SPEC.md), and [PARTS_README.md](PARTS_README.md) for the bundle format, the socket, and the live parts inbox.

## Repository layout

| path | what |
|---|---|
| `seed_v4.py` | trainer, model, and parts format (the engine) |
| `seed_v3_gpu.py` | data-pipeline helper imported by `seed_v4.py` (not the current trainer) |
| `seed_organs.py` | organ bundles, sockets, signing, rehearsal caches |
| `parts_tool.py` | export, harvest, train, import, verify, and publish organs |
| `chat.py` | talk to a checkpoint or a tree plus organs |
| `prep_data.py` | build a tokenizer and `train.bin` / `valid.bin` |
| `ed25519_min.py` | signing helper used by organ bundles |
| `make_task.py` | tiny example task generator for organ tests |
| `plot.py` | loss / growth chart for a run (optional; needs matplotlib) |
| `tokenizers/seed-bpe4096-v1/` | the shared 4,096-token byte-level BPE |
| `socket/seed-socket-v1/` | socket metadata and calibration set (see note below) |
| `scripts/run_seed_v4.ps1` | configurable Windows launcher |

## Socket note

The official `seed-socket-v1` frozen anchor (`anchor_emb.safetensors`, about 6.0 MB, sha prefix `a8438384b2c1b973`) is included in `socket/seed-socket-v1/` along with `socket.json` and the small calibration set. A checkout is enough to join the v1 organ ecosystem.

To start your own socket, run `parts_tool.py make-socket`. Organs are not compatible across different anchors or tokenizer versions.

## License

Apache License 2.0. Copyright 2026 Colby Mcgarrah.

## Support

If Seed is useful to you, donations help keep development going: Cash App [$ColbyMcgarrah0](https://cash.app/$ColbyMcgarrah0)
