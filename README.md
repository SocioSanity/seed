# Seed

A tiny self-growing language model with a **Seed Pool** of plug-and-play **organs** on a standard socket (`seed-socket-v1`).

An **organ** is a whole frozen seed, trained on one skill or tool, then added to a pool. A small trained router mixes the active organs on every token — there is no hand-off. Organs can be hot-added or removed. The goal is capable, customizable AI on small computers — not a giant general model.

Inside a seed, the older per-column units (one cell in every layer, plus a rank-8 adapter, shipped as a `seed-organ/1.1` bundle with a rehearsal cache) are **organelles**. The on-disk format and filenames are unchanged so existing loaders still work. Transplanting organelles with `parts_tool.py import` is experimental and is known to fail when the donor is a whole seed; the pool is the supported way to combine organs. See [ORGANS_README.md](ORGANS_README.md) and [SOCKET_SPEC.md](SOCKET_SPEC.md).

## How it works

- A **seed** is a small growing transformer (Seed v4). Width is a whole number of organelles.
- An **organelle** is one column: one cell in every layer, plus a rank-8 adapter, on a shared frozen embedding anchor.
- Specialists are trained on one skill or tool. The finished seed can be added to a pool as an organ, or an organelle can be harvested with `parts_tool.py` into a signed `seed-organ` bundle (an organelle bundle).
- Organelle bundles use the same socket and tokenizer (`seed-bpe4096-v1`). A rehearsal cache of hard examples lives inside the bundle so later training does not erode the skill.
- A **pool** keeps organs (whole frozen seeds) under a 2 GB weights budget and weights them per token. Every active organ runs on every token. Only the organs you add sit in memory, so a small PC can run a capable, customized AI.

## Vision

Named organs, used as needed for the task at hand. A pool can hold trained seeds and route to them per token.

Colby's starter organs are free. Anyone can train and share their own organs. The engine and the socket spec are open source.

## Hardware

Built and trained on a desktop with an **NVIDIA GTX 1660 Ti (6 GB)**. Training and inference also run on CPU (slower).

## Current results

**run3** (fluent trunk, TinyStories) — starter organ:

- 4.43M parameters (shape 4×2), step 74,300
- Best TinyStories EMA loss 1.5627 = 0.570 bits/byte
- Frozen seed: [`organs/seeds/run3-trunk-final`](organs/seeds/run3-trunk-final.json)

**run4** (tools + reasoning) — starter organ:

- 3.72M parameters (shape 6×1), step 19,822
- Tool/chat val 1.000
- Frozen seed: [`organs/seeds/run4-tools-reasoning-final`](organs/seeds/run4-tools-reasoning-final.json)
- Organelle bundle (column + rehearsal cache): [`organs/run4-tools-reasoning-1.0.1`](organs/run4-tools-reasoning-1.0.1.json)

**Seed Pool v1** (run3 + run4 as organs, trained router):

The all-active pool keeps run4's tool skill and run3's story fluency with no hand-off. On TinyStories, pool 0.533 vs run3 0.536 bits/byte is a tie (no loss of run3's fluency), not an improvement. The router (403k params, 1.5 MB) trained in about 2 minutes and, on its own, leans on run4 for thinking, tool calls and math, and on run3 for stories.

The pool is about 1/12 the weight of the 100M comparison model (16.3 MB vs 199.4 MB fp16) and is faster on both GPU (68 vs 41 tok/s) and CPU (42 vs 16). That 100M checkpoint is only at step 1,350 of 3,662 planned, so it is undertrained. Any "close to the 100M" reading is against this early baseline. Future comparisons will use Qwen 3B coder.

| | run3 | run4 | pool1 | tiny100m (step 1,350) |
|---|---|---|---|---|
| params | 4.43M | 3.72M | 8.55M | 104.5M |
| weights MB fp32 / fp16 | 16.9 / 8.5 | 14.2 / 7.1 | 32.6 / 16.3 | 398.8 / 199.4 |
| tok/s GPU / CPU (4 threads) | 165 / 138 | 133 / 135 | 68 / 42 | 41 / 16 |
| bpb wiki_tool | 4.489 | 0.770 | 0.780 | 0.666 |
| bpb glaive | 4.082 | 0.734 | 0.741 | 0.636 |
| bpb instruct | 3.543 | 1.215 | 1.227 | 1.123 |
| bpb calc | 5.352 | 0.204 | 0.229 | 0.106 |
| bpb tools_misc | 3.511 | 0.502 | 0.423 | 0.330 |
| bpb gsm8k | 3.920 | 0.821 | 0.841 | 0.743 |
| bpb TinyStories | 0.536 | 1.031 | 0.533 | 0.547 |
| tool calls / valid JSON | 0 / - | 23 / 100% | 23 / 91% | 22 / 100% |
| facts: correct queries / right answers (of 20) | 0 / 0 | 16 / 11 | 16 / 9 | 13 / 12 |
| math ok (of 10) | 0 | 4 | 4 | 2 |
| fluency by eye | good | poor | good | good |

Bits per byte are on the seed held-out set (lower is better). The 20-item facts and 10-item math rows are small and noisy (about ±1 run to run). On facts, the pool had 16 correct queries and 9 right answers (run4: 16 and 11; the 100M: 13 and 12).

Two weak spots: short answers after a Wikipedia lookup often paste the wrong part of the article (a run4 limit, not the router's), and 2 of 23 pool tool calls had broken JSON. A seed trained to answer from search results is the intended fix for the first.

**Router behaviour** (mean weight on run4, functional suite):

| task | region | tokens | weight on run4 |
|---|---|---|---|
| fact | think | 393 | 0.9996 |
| fact | tool_call | 922 | 0.9999 |
| fact | tool_result | 6,824 | 0.991 |
| fact | answer | 1,737 | 0.998 |
| math | think / tool_call / answer | 990 | 0.999–1.0 |
| story | story | 203 | 0.138 (10% of tokens > 0.5) |

On training val: TinyStories → run3 0.97; tool text → run4 0.97. The router switches per token, e.g. a story prompt inside the chat format starts on run4 and moves to run3.

Hot add/remove: an older seed the router had never seen was added in 0.3 s and started helping with stories; removing it left the pool's log-probs unchanged.

Full write-up, including the failed-merge numbers and more router detail: [pool1/results/SUMMARY.md](pool1/results/SUMMARY.md).

**Failed merge (merged1):** installing run4's only column into the run3 trunk with `parts_tool.py import` kept story fluency (TinyStories 0.537 vs run3 0.536) but collapsed tool skill: 0/20 correct queries, 1/20 right answers, 1/10 math, 59% valid JSON, and garbled queries. 8.00M params, 56 tok/s on GPU. That transplant path is experimental and is known to fail for whole-seed merges. The pool is what actually combines these organs. Numbers and bpb rows are in the [SUMMARY](pool1/results/SUMMARY.md#failed-merge-merged1).

## Quickstart

### Install

Python 3.10+ (developed on 3.12).

```bash
pip install -r requirements.txt
```

For GPU training, install a CUDA build of PyTorch from [pytorch.org](https://pytorch.org) (this project was trained with `torch 2.6.0+cu124`). The `requirements.txt` pin is CPU-agnostic; swap in the CUDA wheel if you have a GPU.

### Chat with a starter organ

```bash
python chat.py organs/seeds/run3-trunk-final.json
python chat.py organs/seeds/run4-tools-reasoning-final.json --prompt "Once upon a time" --n 120
```

`chat.py` also still accepts a training checkpoint (`latest.pt`). Generation loads the brace decode prior by default; pass `--no-decode-prior` to turn it off.

### Seed Pool

The published registry and router live in `pool1/`. Paths in the registry are relative to the repo root. Override the pool directory with `--dir` or the `SEED_POOL_DIR` environment variable.

```bash
python seed_pool.py list
python seed_pool.py chat --prompt "Once upon a time"
python seed_pool.py chat --tools --prompt "Who was Marie Curie?"
python seed_pool.py add NAME path/to/seed.json
python seed_pool.py remove NAME
```

`--tools` uses the tool helpers in `comparison/tiny100m` (calculator and `read_file`; Wikipedia search is optional if you have a local index).

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

### Harvest and transplant organelles (experimental)

`parts_tool.py import` can install an organelle bundle into a trunk. This path is experimental. It is known to fail when the donor is a whole seed (see merged1 above). Prefer `seed_pool.py` to combine organs.

```bash
# Publish a socket once from a trunk (needed for compatible organelles)
python parts_tool.py make-socket --from runs/run1/latest.pt --socket-dir socket/seed-socket-v1

# Train a specialist organelle on a task
python parts_tool.py train-organ --host runs/specialist/latest.pt --task task.jsonl --task-eval task_eval.jsonl --name wiki-search --out organs/my/wiki-search-1.0.0

# Or harvest an organelle already inside a trained seed
python parts_tool.py harvest --host runs/specialist/latest.pt --organ o1 --out organs/my/wiki-search-1.0.0 --name wiki-search

# Experimental: transplant the starter run4 organelle into a trunk
python parts_tool.py import organs/run4-tools-reasoning-1.0.1 --host runs/trunk/latest.pt --out runs/merged.pt
```

Command names (`train-organ`, `--organ`, format `seed-organ/1.1`) are unchanged so existing scripts and loaders keep working. See [ORGANS_README.md](ORGANS_README.md), [SOCKET_SPEC.md](SOCKET_SPEC.md), and [PARTS_README.md](PARTS_README.md) for the bundle format, the socket, and the live parts inbox.

## Decode prior

The official `seed-socket-v1` frozen anchor was trained on TinyStories only, so about 183 tokens never seen there (`{`, `}`, `=`, `_`, `[`, and most raw bytes) have nearly identical embeddings. Socket seeds get the group's total probability right but split it evenly, so `{` could come out as a random byte (shown as U+FFFD).

`tokenizers/seed-bpe4096-v1/cluster_prior.pt` re-splits probability inside that group by bigram frequency. `load_model_any` and the pool attach it automatically; `chat.py --no-decode-prior` turns it off. Probability outside the group is untouched. The prior is a decode-side fix; a later socket with an anchor trained on code and JSON is the real fix.

The prior was built by `scripts/build_decode_prior.py`.

## 100M comparison model

A 104.5M-parameter model (`tiny100m`, best EMA at step 1,350 of 3,662 planned) is **comparison-only** — not a Seed, and not the product. It is undertrained; later comparisons will use Qwen 3B coder. Its weights are about 200 MB in fp16, so they ship as a GitHub Release asset rather than a repo file.

Get them from the [v0.2.0 release](https://github.com/SocioSanity/seed/releases/tag/v0.2.0):

- `tiny100m-best-step1350-fp16.safetensors` (the weights)
- `tiny100m-compare.zip` (config, tokenizer, and a small loader)

Put the weights next to the files in [`comparison/tiny100m/`](comparison/tiny100m/) (already in this repo) and load:

```bash
python comparison/tiny100m/load_tiny100m.py --weights tiny100m-best-step1350-fp16.safetensors --prompt "Once upon a time"
```

See [comparison/tiny100m/README.md](comparison/tiny100m/README.md) for the layout and how the loader works. The Seed pool runs on the GPU; this comparison model is a separate checkpoint.

## Repository layout

| path | what |
|---|---|
| `seed_v4.py` | trainer, model, and parts format (the engine) |
| `seed_v3_gpu.py` | data-pipeline helper imported by `seed_v4.py` (not the current trainer) |
| `seed_organs.py` | organelle bundles, sockets, signing, rehearsal caches |
| `parts_tool.py` | export, harvest, train, import, verify, and publish organelles (import is experimental) |
| `seed_pool.py` | Seed Pool v1: router over organs (whole frozen seeds) |
| `chat.py` | talk to a checkpoint, a tree, or a seed plus organelles |
| `prep_data.py` | build a tokenizer and `train.bin` / `valid.bin` |
| `ed25519_min.py` | signing helper used by organelle bundles |
| `make_task.py` | tiny example task generator for organelle tests |
| `plot.py` | loss / growth chart for a run (optional; needs matplotlib) |
| `organs/seeds/` | starter organs: run3 trunk and run4 tools+reasoning, fp16 tree parts |
| `organs/run4-tools-reasoning-1.0.1` | starter organelle bundle (run4 column + rehearsal cache) |
| `pool1/` | trained pool router, registry, and [results / SUMMARY](pool1/results/SUMMARY.md) |
| `tokenizers/seed-bpe4096-v1/` | the shared 4,096-token byte-level BPE, plus the decode prior |
| `socket/seed-socket-v1/` | socket metadata and calibration set (see note below) |
| `comparison/tiny100m/` | loader and docs for the 100M comparison model (weights on the release) |
| `scripts/run_seed_v4.ps1` | configurable Windows launcher |

## Socket note

The official `seed-socket-v1` frozen anchor (`anchor_emb.safetensors`, about 6.0 MB, sha prefix `a8438384b2c1b973`) is included in `socket/seed-socket-v1/` along with `socket.json` and the small calibration set. A checkout is enough to join the v1 organelle ecosystem.

To start your own socket, run `parts_tool.py make-socket`. Organelles are not compatible across different anchors or tokenizer versions.

## License

Apache License 2.0. Copyright 2026 Colby Mcgarrah.

## Support

If Seed is useful to you, donations help keep development going: Cash App [$ColbyMcgarrah0](https://cash.app/$ColbyMcgarrah0)
