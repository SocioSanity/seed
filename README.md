# Seed

A tiny self-growing language model with a **Seed Pool** of plug-and-play **organs** on a standard socket (`seed-socket-v1`).

An **organ** is a whole frozen seed, trained on one skill or tool, then added to a pool. A small trained router mixes the active organs on every token — there is no hand-off. Organs can be hot-added or removed. The goal is capable, customizable AI on small computers — not a giant general model.

Inside a seed, the older per-column units (one cell in every layer, plus a rank-8 adapter, shipped as a `seed-organ/1.1` bundle with a rehearsal cache) are **organelles**. The on-disk format and filenames are unchanged so existing loaders still work. Transplanting organelles with `parts_tool.py import` is experimental and is known to fail when the donor is a whole seed; the pool is the supported way to combine organs. See [ORGANS_README.md](ORGANS_README.md) and [SOCKET_SPEC.md](SOCKET_SPEC.md).

---

**Grow your own seed.** A complete beginner-friendly walkthrough — preparing data, launching the trainer, monitoring a run, pausing and resuming, exporting, and sharing a specialist — is in **[GROW_YOUR_OWN.md](GROW_YOUR_OWN.md)**. Start with Option A (Seed v2.0, 150k tokenizer) unless you need a seed that plugs into the published Pool 5 router.

---

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

### Seed Pool 5 (v0.4.0): 16 specialist organs and a looping router

Pool 5 is a 16-organ pool of frozen Seed v4 specialists on the shared `seed-bpe4096-v1` tokenizer and `seed-socket-v1` anchor. Together they are about **88.7M parameters** and **338.5 MB** of fp32 weights (about 169 MB in the published fp16 organs). That still fits comfortably under the **2 GB weight cap** on a GTX 1660 Ti.

The seeds:

| organ | params | shape | what it is for |
|---|---|---|---|
| `run3` | 4.43M | 4×2 | fluent English / TinyStories trunk |
| `run4` | 3.72M | 6×1 | tools + reasoning (`wiki_search`, `calculator`) |
| `json1` | 4.43M | 4×2 | JSON and tool-call argument filling |
| `answer1` | 4.43M | 4×2 | short answers from `<tool_result>` text |
| `math1` | 4.43M | 4×2 | math word problems → plan + calculator calls |
| `reason1` | 4.43M | 4×2 | step-by-step puzzles and GSM8K-style chains |
| `plan1` | 4.43M | 4×2 | word-problem planner |
| `plan2` | 10.14M | 8×3 | larger word-problem planner |
| `py1` | 10.14M | 8×3 | Python writer |
| `fix1` | 5.86M | 6×2 | run-and-fix with `python_exec` |
| `inst1` | 5.86M | 6×2 | instruction following |
| `c1` | 4.43M | 4×2 | C writer |
| `js1` | 4.43M | 4×2 | JavaScript writer |
| `sh1` | 5.86M | 6×2 | Bash / shell writer |
| `ps1` | 5.86M | 6×2 | PowerShell writer |
| `alg1` | 5.86M | 6×2 | language-agnostic algorithmic reasoning |

A confidence-gated **1–3 pass looping router** picks a sparse active set per token, then takes another pass (adding more organs) only when it is not confident. Three variants were trained on the same 16 seeds. The winner is the **single-head Top-6** router (`pool5/`, `init_k=6`, `step_k=3`).

| variant | macro BPB (lower is better) | greedy tok/s (GPU) | coding eval (20 Python tasks) |
|---|---|---|---|
| **single-head Top-6 (winner)** | **0.8293** | **17.2** | 0/20 chat, 0/20 complete |
| single-head Top-3 | 0.8375 | 26.7 | 0/20 |
| multi-head Top-3 (3 heads) | 0.8959 | 19.4 | 0/20 |

Top-6 domain bits per byte (held-out, lower is better):

| python | C | JS | bash | PowerShell | math | reason | alg (real / synth) | Colby texts |
|---|---|---|---|---|---|---|---|---|
| 0.447 | 0.649 | 1.042 | 1.118 | 0.809 | 0.174 | 0.075 | 1.628 / 1.376 | 0.976 |

What worked: the pool is fast enough for interactive use on a 1660 Ti, English is fluent (the run3 trunk is still in the mix), and domain BPB is strong on math, reasoning, and the code/shell languages the specialists were trained on.

What we diagnosed honestly: on a **strict 20-task Python coding eval**, both the Top-6 pool and the `py1` organ alone scored **0/20**. The 4,096-token BPE, the ~4–10M per-seed size cap, and router prose bleed (non-Python organs still getting weight on coding prompts) are the bottleneck we are targeting next — not a claim that the architecture failed. See [pool5/results/SUMMARY.md](pool5/results/SUMMARY.md).

```bash
python seed_pool.py --dir pool5 list
python seed_pool.py --dir pool5 chat --tools --prompt "What is 17 * 19?"
```

### Coming soon: Seed v2.0

Seed v2.0 starts from that diagnosis. A new custom **150,000-vocab** code-and-English ByteLevel BPE is already in the repo at [`tokenizers/seed-bpe150k-v2/`](tokenizers/seed-bpe150k-v2/): single-digit splitting, intact indent tokens, and 13 chat / tool / think special tokens. The plan is a **shared, factorized embedding** so a 150k vocab does not eat the whole budget, then dynamically grown specialist seeds scaling toward a **~500M-parameter pool** that still fits under the **2 GB weight cap** on a GTX 1660 Ti — starting with English fluency training to a validation loss under **1.0**.

### Seed Pool v2 (v0.3.0): two specialist organs + a fair Qwen comparison

v0.3.0 adds two small specialist organs trained from scratch on the same tokenizer and socket anchor, and hot-adds them to a copy of the pool (`pool2/`, 4 organs, router retrained for 900 steps). The comparison baseline is now **Qwen2.5-Coder-3B** instead of the undertrained 100M.

- **json1** (4.43M, 4×2, step 4,750): tool-call argument filling and reading, writing and fixing JSON — [`organs/seeds/json1-final`](organs/seeds/json1-final.json)
- **answer1** (4.43M, 4×2, step 4,500): writes a short final answer from `<tool_result>` text — [`organs/seeds/answer1-final`](organs/seeds/answer1-final.json)
- Data recipes: [`recipes/`](recipes/)

All rows below use greedy decoding and one test run per model. Qwen is the `qwen2.5-coder:3b` Q4_K_M GGUF (llama.cpp) for the functional tests; its bits per byte come from the official HF weights in fp32 (the Q4_K_M file was not scored).

| | pool1 | pool2 | Qwen2.5-Coder-3B, native tool template | Qwen, no tools |
|---|---|---|---|---|
| params / weights | 8.55M / 32.6 MB fp32 | 17.4M / 66.4 MB fp32 (33.2 fp16) | 3.09B / 1,840 MB (Q4_K_M) | |
| tok/s GPU / CPU (4 threads) | 56 / 25 | 29 / 14 | 86 / 6 | |
| 20 common facts, right answers | 11 | 16 | 20 | 20 |
| 15 obscure facts (answers only in the local Wikipedia index) | 7 | 12 | 12 | 2 |
| 10 math | 5 | 5 | 8 | 8 |
| valid JSON tool calls | 100% (38) | 100% (39) | 100% (88; none use the `<tool_call>` tag) | – |
| bpb wiki_tool (held out, our format) | 0.780 | **0.716** | 0.751 | |
| bpb glaive (held out, neutral tool-chat) | 0.741 | 0.670 | **0.384** | |
| bpb TinyStories | 0.533 | **0.533** | 0.576 | |

Honest caveats:

- The wiki_tool documents are in Seed's own tool-transcript format, which favours the seeds. **Glaive is the neutral test, and Qwen is far better there** (0.384 vs 0.670).
- Qwen only uses tools through its own chat template. With Seed's format plus a schema and 3 worked examples it ends its turn after `<think>` and makes 0 calls (format sensitivity). With its native template it makes correct calls every time but re-issues the same call instead of answering; 43 of 45 questions needed an extra "now answer" turn that the seeds did not get.
- pool2's answers mostly copy the first sentence of the right article, sometimes garbled; Qwen's are fluent rewrites. Answer checks are keyword-based.
- The 15 obscure questions are entities whose titles appear in no training file, with hand-picked answer keys. Small sets: treat ±1 as noise.
- pool2 runs all four organs on every token, so it is the slowest pool (29 tok/s on GPU).

Router (pool2): json1 carries tool calls (about 0.6–0.7 of the weight), answer1 reads tool results (about 0.75), run4 leads on reasoning (0.72 on math thinking), and run3 on stories (0.90). Full write-up, prompts and per-item results: [pool2/results/SUMMARY.md](pool2/results/SUMMARY.md), [pool2/results/qwen_prompts.md](pool2/results/qwen_prompts.md).

### Seed Pool v1 (v0.2.0)

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

The pool is about 1/12 the weight of the 100M comparison model (16.3 MB vs 199.4 MB fp16) and is faster on both GPU (68 vs 41 tok/s) and CPU (42 vs 16). That 100M checkpoint is only at step 1,350 of 3,662 planned, so it is undertrained. Any "close to the 100M" reading is against this early baseline. From v0.3.0 the comparison baseline is Qwen2.5-Coder-3B (above).

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

The published registries and routers live in `pool1/` (2 organs), `pool2/` (4 organs), and `pool5/` (16 organs, looping Top-6 winner). Paths in the registry are relative to the repo root. Override the pool directory with `--dir` or the `SEED_POOL_DIR` environment variable.

```bash
python seed_pool.py --dir pool5 list
python seed_pool.py --dir pool5 chat --tools --prompt "What is 17 * 19?"
python seed_pool.py --dir pool2 chat --tools --prompt "What is German submarine U-454?"
python seed_pool.py --dir pool1 chat --prompt "Once upon a time"
python seed_pool.py --dir pool5 add NAME path/to/seed.json
python seed_pool.py --dir pool5 remove NAME
```

`--tools` uses the runtimes in [`tools/`](tools/README.md): `calculator`, `read_file`, `python_exec`, and optional `wiki_search` (set `SEED_WIKI` to a local introductions index).

### Grow your own seed (single consumer GPU)

The full step-by-step guide (data, flags, export, router wiring, and sharing rules) is **[GROW_YOUR_OWN.md](GROW_YOUR_OWN.md)**.

This is the supported path for training a new specialist and adding it to a pool. The dynamic-growth trainer is `seed_v4.py`. It starts small (2 layers × 1 cell by default) and grows cells and layers automatically under a weights cap and a VRAM cap. The defaults fit a **GTX 1660 Ti (6 GB)**; they also run on CPU (slower).

**Use the official tokenizer and socket** if you want the new seed to plug into the published pool. Training a fresh BPE with `prep_data.py` (no `--tokenizer`) produces a seed that will not load on `seed-socket-v1`.

```bash
# 1) Install (Python 3.10+; developed on 3.12)
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
# If you have an NVIDIA GPU, install a CUDA build of PyTorch from https://pytorch.org
# (this project was trained with torch 2.6.0+cu124).

# 2) Tokenize your text with the official 4,096-token tokenizer
python prep_data.py --data-dir path/to/your_texts --out data \
    --tokenizer tokenizers/seed-bpe4096-v1/tokenizer.json
# Or, from TinyStories files:
# python prep_data.py --train-text TinyStoriesV2-GPT4-train.txt \
#     --valid-text TinyStoriesV2-GPT4-valid.txt --out data \
#     --tokenizer tokenizers/seed-bpe4096-v1/tokenizer.json

# 3) Grow the seed. Caps keep weights under 2 GB and process VRAM around 5 GB.
#    Add a specialty corpus with --mix (plain .txt). Use --resume to continue later.
python seed_v4.py --out runs/myseed --device cuda --resume --data data \
    --tokenizer-dir tokenizers/seed-bpe4096-v1 \
    --socket-dir socket/seed-socket-v1 \
    --vram-cap-mb 5000 --mem-cap-mb 2048 \
    --minutes 1440 --mix path/to/specialty.txt

# Windows helper with the same defaults:
# powershell -ExecutionPolicy Bypass -File scripts/run_seed_v4.ps1 -Out runs\myseed -Minutes 1440 -Device cuda -Data data -Mix path\to\specialty.txt

# CPU instead of GPU:
# python seed_v4.py --out runs/myseed --device cpu --threads 3 --minutes 180 --data data \
#     --tokenizer-dir tokenizers/seed-bpe4096-v1 --socket-dir socket/seed-socket-v1
```

Stop a run cleanly by creating the file `runs/myseed/STOP` (it saves, then exits). Resume with `--resume` and the same `--out`. The trainer writes `runs/myseed/latest.pt` and `train.log` as it goes.

```bash
# 4) Export a shareable fp16 organ (weights only, no optimizer; typically 8-20 MB)
python scripts/export_seed.py runs/myseed/latest.pt organs/seeds/myseed \
    --name myseed --description "what this seed learned"

# 5) Add it to a pool. Seeds stay frozen; only the small router is trained.
python seed_pool.py --dir pool5 add myseed organs/seeds/myseed.json --desc "what this seed learned"
python seed_pool.py --dir pool5 calibrate myseed --steps 200
# Or retrain the dense (all-active) router from seed_pool.py:
# python seed_pool.py --dir pool5 train --max-steps 900

# 6) Chat
python seed_pool.py --dir pool5 chat --tools --prompt "Your question"
```

The looping router used by Pool 5 lives in `pool5/eval/pool5_loop.py`. Retraining it needs your own mix of domain text; see [pool5/results/SUMMARY.md](pool5/results/SUMMARY.md).

### Link seeds and route them with looped Top-6

Pool 5's published router is a **confidence-gated looping Top-6** (`--mode single --init-k 6 --step-k 3`). Use this whenever you want the winning architecture, including after you drop in a new seed.

```bash
# 1) Put organs in a pool registry (paths are relative to the repo root)
python scripts/pool_register.py --dir pool5 list
python scripts/pool_register.py --dir pool5 add myseed organs/seeds/myseed.json --desc "what it learned"
# or: python seed_pool.py --dir pool5 add myseed organs/seeds/myseed.json --desc "what it learned"

# 2) Retrain the looping router on the new seed set
#    (required after add/remove: router keys are per-name)
#    Copy pool5/mix.example.json and point each path at your local .txt files.
python pool5/eval/pool5_loop.py base --dir pool5 --mode single --init-k 6 --step-k 3 --mix-json pool5/mix.json
python pool5/eval/pool5_loop.py loop --dir pool5 --minutes 15 --mix-json pool5/mix.json

# 3) Chat with the looped Top-6 router (greedy, tau=0.9, tools on)
python pool5/eval/pool5_loop.py chat --dir pool5 --tools --prompt "What is 17 * 19?"
```

`base` trains the sparse context router (seeds stay frozen). `loop` trains the 1–3 pass confidence head on top (the base router stays frozen). The published winner weights are `pool5/router_base.pt` and `pool5/router_loop.pt`. Other variants from the same 16 seeds are under `pool5/variants/`.

### Grow your own seed (v2.0, 150k tokenizer)

The full step-by-step guide is **[GROW_YOUR_OWN.md](GROW_YOUR_OWN.md)** (Option A).

Seed v2.0 is a new trainer for the official **150,000-vocab** ByteLevel BPE. It uses a **factorized, tied embedding** (`--e-dim 128`, table is `150000 × 128`) and residual width `--d 512`. It starts at 4 layers × 2 cells and grows under the same 2 GB weight cap. It is **not** compatible with Pool 5 or `seed-socket-v1` — those stay on the 4k tokenizer.

The official tokenizer is at [`tokenizers/seed-bpe150k-v2/`](tokenizers/seed-bpe150k-v2/). `tokenizer.json` SHA-256 is `fc67f00b300e872734502b96eeee00440be6266bfe83338a164acfb7ca847d73`. Do not train a new 150k BPE if you want to share v2.0 seeds.

```bash
# 1) Copy the example mix and point every "path" (and val_path) at your own local .txt files.
#    The listed sources are public datasets (FineWeb-Edu, Wikipedia, SmolTalk, TinyStories,
#    Project Gutenberg). The files themselves are not in the repo.
cp v2/mix.example.json v2/mix.json
# then edit v2/mix.json

# 2) Grow an English (or specialist) seed. Stops when mixed val CE < 1.0 (default --target-ce).
#    Works on a 1660 Ti (6 GB) with the caps below; a 4090 or H100 can raise --batch / --ctx.
python v2/seed_v2.py --tokenizer tokenizers/seed-bpe150k-v2/tokenizer.json \
    --mix-json v2/mix.json --out runs/v2-eng1 --device cuda --resume \
    --d 512 --e-dim 128 --target-ce 1.0 \
    --vram-cap-mb 4800 --mem-cap-mb 2048

# Windows helper:
# powershell -ExecutionPolicy Bypass -File scripts/run_seed_v2.ps1 -Out runs\v2-eng1 -MixJson v2\mix.json

# 3) Stop cleanly with runs/v2-eng1/STOP (saves, then exits). Resume with the same --out and --resume.
```

See [v2/README.md](v2/README.md) and [tokenizers/seed-bpe150k-v2/](tokenizers/seed-bpe150k-v2/).


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
| [`GROW_YOUR_OWN.md`](GROW_YOUR_OWN.md) | step-by-step guide to train, export, and share your own seeds (v2.0 and Pool 5) |
| `seed_v4.py` | dynamic-growth trainer, model, and parts format (the engine; includes the GPU optimizer-device fix) |
| `seed_v3_gpu.py` | data-pipeline helper imported by `seed_v4.py` (not the current trainer) |
| `seed_organs.py` | organelle bundles, sockets, signing, rehearsal caches |
| `parts_tool.py` | export, harvest, train, import, verify, and publish organelles (import is experimental) |
| `seed_pool.py` | Seed Pool: dense router over organs (whole frozen seeds); train / calibrate / chat |
| `tools/` | tool runtimes: calculator, wiki_search, read_file, python_exec — [README](tools/README.md) |
| `chat.py` | talk to a checkpoint, a tree, or a seed plus organelles |
| `prep_data.py` | build `train.bin` / `valid.bin` (pass `--tokenizer` to use the official 4k vocab) |
| `scripts/export_seed.py` | export a v4 checkpoint as a shareable fp16 organ |
| `scripts/pool_register.py` | add / list / remove seeds in a pool `registry.json` |
| `scripts/run_seed_v4.ps1` | configurable Windows launcher for the v4 trainer |
| `scripts/run_seed_v2.ps1` | Windows launcher for the v2.0 150k trainer |
| `v2/seed_v2.py` | Seed v2.0 dynamic-growth trainer (150k tokenizer, factorized embeddings) |
| `v2/mix.example.json` | example English mix; point the paths at your own local `.txt` files |
| `scripts/build_decode_prior.py` | builds the brace decode prior |
| `ed25519_min.py` | signing helper used by organelle bundles |
| `make_task.py` | tiny example task generator for organelle tests |
| `plot.py` | loss / growth chart for a run (optional; needs matplotlib) |
| `organs/seeds/` | free organs: the Pool 5 set (16 fp16 tree parts) plus earlier starters |
| `organs/run4-tools-reasoning-1.0.1` | starter organelle bundle (run4 column + rehearsal cache) |
| `pool1/` | 2-organ pool: router, registry, [results / SUMMARY](pool1/results/SUMMARY.md) |
| `pool2/` | 4-organ pool: router, registry, eval scripts, [results / SUMMARY](pool2/results/SUMMARY.md) |
| `pool5/` | 16-organ pool: looping Top-6 winner (`router_base.pt` + `router_loop.pt`), variants, [results / SUMMARY](pool5/results/SUMMARY.md) |
| `pool5/eval/pool5_loop.py` | train and chat with the confidence-gated looping Top-6 router |
| `recipes/` | data builders and training args for the specialist organs |
| `tokenizers/seed-bpe4096-v1/` | the shared 4,096-token byte-level BPE, plus the decode prior |
| `tokenizers/seed-bpe150k-v2/` | Seed v2.0 150k tokenizer (coming soon; not used by Pool 5) |
| `socket/seed-socket-v1/` | socket metadata and calibration set (see note below) |
| `comparison/tiny100m/` | loader and docs for the 100M comparison model (weights on the release) |

## Socket note

The official `seed-socket-v1` frozen anchor (`anchor_emb.safetensors`, about 6.0 MB, sha prefix `a8438384b2c1b973`) is included in `socket/seed-socket-v1/` along with `socket.json` and the small calibration set. A checkout is enough to join the v1 organelle ecosystem.

To start your own socket, run `parts_tool.py make-socket`. Organelles are not compatible across different anchors or tokenizer versions.

## License

Apache License 2.0. Copyright 2026 Colby Mcgarrah.

## Support

If Seed is useful to you, donations help keep development going: Cash App [$ColbyMcgarrah0](https://cash.app/$ColbyMcgarrah0)
