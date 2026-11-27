# Grow Your Own Seed

This is the step-by-step guide for training a Seed specialist on your own machine and, if you want, sharing it with the community.

There are **two separate lines**. They use different tokenizers and are **not compatible** with each other:

| | Option A (recommended) | Option B |
|---|---|---|
| Trainer | `v2/seed_v2.py` | `seed_v4.py` |
| Tokenizer | `tokenizers/seed-bpe150k-v2/` (150,000 vocab) | `tokenizers/seed-bpe4096-v1/` (4,096 vocab) |
| Socket | Shared factorized embedding (`--e-dim 128`, `--d 512`) | `socket/seed-socket-v1` |
| Joins | Future Seed v2.0 specialist pool | Published **Pool 5** looped Top-6 router |

If you are starting fresh, use **Option A**. Use **Option B** only when you want a seed that plugs into the published 16-organ Pool 5.

All commands below are meant to be run from the **repository root** after you `cd` into the clone.

---

## 1. Prerequisites & Setup

You will need:

- **Python 3.10–3.12** (developed on 3.12)
- An NVIDIA GPU with a CUDA build of PyTorch if you want reasonable training speed. The project was trained on a **GTX 1660 Ti (6 GB)** with `torch 2.6.0+cu124`. Training also runs on CPU (`--device cpu`), but it is much slower.
- A few gigabytes of disk for datasets and checkpoints
- Training text you are allowed to use (permissive / commercial-safe licenses only if you plan to share a seed — see [Compatibility Rules](#4-compatibility-rules-for-sharing-seeds))

### Clone and install

```bash
git clone https://github.com/SocioSanity/seed.git
cd seed

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt` installs `torch>=2.0`, `numpy`, `tokenizers`, `safetensors`, and `psutil`. The PyTorch pin is CPU-agnostic. **If you have an NVIDIA GPU, install a CUDA build of PyTorch from [pytorch.org](https://pytorch.org)** after (or instead of) the CPU wheel, matching your driver. This project was trained with `torch 2.6.0+cu124`.

Optional extras in the same file: `pypdf` (so `prep_data.py --data-dir` can read PDFs) and `matplotlib` (for `plot.py` charts).

Check that GPU PyTorch actually sees your card:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

You want `True` and your GPU name if you plan to train with `--device cuda`.

---

## 2. Option A (Recommended): Grow a Seed v2.0 Specialist (150k Tokenizer)

Seed v2.0 is a new dynamic-growth trainer built for the official **150,000-vocab** ByteLevel BPE. It uses a **factorized, tied embedding** (`--e-dim 128`, table is `150000 × 128`) and residual width `--d 512`, so every v2.0 specialist can share the same vocab table shape. It starts at **4 layers × 2 cells** and grows cells and layers automatically (up to `--max-layers 24` and `--max-cells 6`, and up to `--max-params`, default 150 million) under a weights cap.

It is **not** compatible with Pool 5, `seed_v4.py`, or `seed-socket-v1`. Those stay on the 4k tokenizer.

The official tokenizer is [`tokenizers/seed-bpe150k-v2/tokenizer.json`](tokenizers/seed-bpe150k-v2/tokenizer.json). Its SHA-256 must be:

```
fc67f00b300e872734502b96eeee00440be6266bfe83338a164acfb7ca847d73
```

Do **not** train a new 150k BPE if you want to share or mix Seed v2.0 specialists. They must all use this exact file. Verify it after you clone:

```bash
# Linux / macOS
sha256sum tokenizers/seed-bpe150k-v2/tokenizer.json

# Windows PowerShell
Get-FileHash tokenizers\seed-bpe150k-v2\tokenizer.json -Algorithm SHA256
```

### 2.1 Prepare your data and a `mix.json`

The trainer streams **plain UTF-8 `.txt` files**. Nothing is pre-tokenized. The files themselves are not in the repo — you point a JSON mix file at paths on your machine.

Copy the official example, then edit the paths:

```bash
# Linux / macOS
cp v2/mix.example.json v2/mix.json

# Windows
copy v2\mix.example.json v2\mix.json
```

`v2/mix.example.json` is a six-source English mix (FineWeb-Edu, Wikipedia, SmolTalk, TinyStories, public-domain philosophy, Project Gutenberg). Those are public datasets; you have to download them yourself and point every `"path"` (and `"val_path"` if present) at your local `.txt` files.

A **minimal, copy-pasteable** mix — one UTF-8 text file, with a separate validation file — looks like this:

```json
[
  {
    "name": "english",
    "path": "datasets/my_english_train.txt",
    "val_path": "datasets/my_english_valid.txt",
    "weight": 1.0,
    "sep": "blank",
    "license": "CC BY-SA 4.0 (say the real license here)"
  }
]
```

You can list as many sources as you want. Each object may use these fields (this is the complete set `v2/seed_v2.py` reads):

| field | required? | what it does |
|---|---|---|
| `name` | yes | label used in logs and per-source validation |
| `path` | yes | UTF-8 `.txt` file, relative to the repo root (or an absolute path) |
| `weight` | yes | relative sampling weight (they are normalized to sum to 1) |
| `val_path` | no | a separate held-out `.txt`. If omitted, the trainer holds out the last `holdout_frac` of `path` |
| `holdout_frac` | no | default `0.01` (1%) when `val_path` is absent |
| `sep` | no | default `"blank"` |
| `license` | no | recorded on the source (please fill this in if you might share the seed) |
| `max_digit_frac` | no | default `0.0` (off). If set, e.g. `0.2`, chunks denser than that fraction of digits are retried — useful for Project Gutenberg number dumps |

How `sep` is applied (verified in `v2/seed_v2.py`):

- Every source first replaces `<|endoftext|>` with the tokenizer's `<eos>` token.
- `"sep": "blank"` additionally treats a blank line (two or more newlines) as a document boundary (`<eos>`).
- Any other value, including the example-file values `"eot"` and `"none"`, does not add extra splitting: `"eot"` is the right choice when your file already uses `<|endoftext|>` between documents; `"none"` leaves the text otherwise unsplit.

A two-source example that holds out 2% of a single file (no separate `val_path`):

```json
[
  {
    "name": "english",
    "path": "datasets/my_english.txt",
    "holdout_frac": 0.02,
    "weight": 0.7,
    "sep": "blank",
    "license": "Public domain (Project Gutenberg)"
  },
  {
    "name": "code",
    "path": "datasets/my_python.txt",
    "val_path": "datasets/my_python_valid.txt",
    "weight": 0.3,
    "sep": "eot",
    "license": "MIT / Apache-2.0 (permissive code only)"
  }
]
```

Put only text you have the right to train on in those files. If you want to contribute the resulting seed, use permissive / commercial-safe data only (see [§4](#4-compatibility-rules-for-sharing-seeds)).

### 2.2 Launch the trainer

From the repository root, this is the canonical command. It uses the official 150k tokenizer, the shared v2.0 socket dimensions, dynamic growth, and the 6 GB card defaults (tested on a GTX 1660 Ti):

```bash
python v2/seed_v2.py --tokenizer tokenizers/seed-bpe150k-v2/tokenizer.json \
    --mix-json v2/mix.json --out runs/v2-myspecialist --device cuda --resume \
    --d 512 --e-dim 128 \
    --start-layers 4 --start-cells 2 --max-layers 24 --max-cells 6 \
    --batch 8 --accum 4 --vram-cap-mb 4800 --mem-cap-mb 2048 \
    --max-params 150000000 --target-ce 1.0
```

On Windows you can use the helper instead (same tokenizer, `--d 512`, `--e-dim 128`, `--target-ce 1.0`, `--vram-cap-mb 4800`, `--mem-cap-mb 2048`, and `--resume`; growth start/max and `--batch` then fall back to the Python defaults above):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_seed_v2.ps1 -Out runs\v2-myspecialist -MixJson v2\mix.json -Device cuda
```

The trainer starts small, grows when validation loss plateaus, writes checkpoints under `--out`, and **stops by itself when mixed validation cross-entropy (`val_ce`, nats/token, raw or EMA, whichever is better) drops below `--target-ce`** (default `1.0`). Bits-per-byte (`val_bpb`) is logged on every eval but is **not** a stop condition.

Default context length is `--ctx 512`. Tokens per optimizer step are `batch × accum × ctx` (with the 6 GB defaults: `8 × 4 × 512 = 16,384`).

CPU instead of GPU:

```bash
python v2/seed_v2.py --tokenizer tokenizers/seed-bpe150k-v2/tokenizer.json \
    --mix-json v2/mix.json --out runs/v2-myspecialist --device cpu --threads 2 --resume \
    --d 512 --e-dim 128 --start-layers 4 --start-cells 2 --max-layers 24 --max-cells 6 \
    --target-ce 1.0
```

### 2.3 Tune `--batch`, `--accum`, `--vram-cap-mb`, and `--max-params` for your GPU

Keep **`--d 512 --e-dim 128`** and the official tokenizer on every v2.0 seed, or they will not share a socket with everyone else's. The knobs you should change for a bigger (or tighter) card are the memory and throughput flags.

`--vram-cap-mb` is a **whole-process GPU cap**. `--mem-cap-mb` is a cap on **model weight size in fp32 bytes** (default `2048` = 2 GB), not on training overhead. The trainer also uses a memory probe plus selective gradient checkpointing (`--grad-ckpt auto` by default) to stay under the VRAM cap.

| GPU memory | Starting point (all of these flags exist on `v2/seed_v2.py`) |
|---|---|
| **6 GB** (GTX 1660 Ti and similar) | Use the defaults: `--batch 8 --accum 4 --ctx 512 --vram-cap-mb 4800 --mem-cap-mb 2048 --max-params 150000000`. This is the combination the trainer was built around. |
| **24 GB** | Keep the shared socket (`--d 512 --e-dim 128`). Raise `--vram-cap-mb` to leave a safety margin under 24 GB (for example `20000`). Raise `--batch` and/or `--ctx` for more tokens per step (this uses more VRAM). Raise `--accum` to grow the effective batch **without** extra VRAM. Raise `--max-params` if you want the seed to grow past the default 150M cap, and raise `--mem-cap-mb` if the fp32 weight budget should grow with it. |
| **80 GB** | Same idea, with more headroom: a `--vram-cap-mb` such as `70000`, a larger `--batch` and/or `--ctx`, and a higher `--max-params` / `--mem-cap-mb` if you want a much larger specialist. |

If you hit a CUDA out-of-memory error, lower `--batch` or `--ctx` first, then rerun with the same `--out` and `--resume`.

### 2.4 Monitor progress

Everything lands in your `--out` directory (for example `runs/v2-myspecialist/`):

| file | what it is |
|---|---|
| `train.log` | human-readable log, including `EVENT` lines and `SAMPLE` generations |
| `metrics.csv` | one row every `--eval-every` steps (default 250) |
| `events.jsonl` | machine-readable growth / immune / target events |
| `samples.txt` | prompt completions, appended every `--save-every` steps (default 1,500) and when the CE target is hit |
| `latest.pt` | full resume checkpoint (weights, EMA, optimizer, RNG, trainer state) |
| `best_ce.pt` | best mixed `val_ce` so far (weights + EMA + metrics; no optimizer) |
| `best_bpb.pt` | best mixed `val_bpb` so far, same format as `best_ce.pt` |

`metrics.csv` columns are:

```
step, minutes, tokens, train_loss, val_ce, val_bpb, val_ce_ema, val_bpb_ema,
per_source_ema, layers, organs, params, lr_mult, phase, decor, balance,
tok_s, step_ms, data_wait_ms, ckpt_layers, hunger, rss_mb, vram_peak_mb
```

Watch `val_ce` and `val_bpb` come down, and `layers` / `organs` / `params` when a growth trial is kept. `per_source_ema` is a JSON object of `(ce, bpb)` per mix source. `hunger` is `well_fed`, `tight`, or `near_cap` — growth pauses when the run is near the memory caps.

Default sample prompts (override with `--prompts`, separated by `|`) are:

```
Once upon a time
The history of the city
User: Why is the sky blue?\nAssistant:
```

On Windows, from another terminal:

```powershell
Get-Content runs\v2-myspecialist\train.log -Wait -Tail 30
```

On Linux / macOS: `tail -f runs/v2-myspecialist/train.log`.

### 2.5 Pause and resume cleanly

To **pause**, create an empty `STOP` file inside the run directory. The trainer notices it at the start of a step, saves `latest.pt`, and exits.

```bash
# Linux / macOS
touch runs/v2-myspecialist/STOP

# Windows PowerShell
New-Item -ItemType File -Force runs\v2-myspecialist\STOP
```

Ctrl+C also triggers a save and a clean exit.

To **resume**, delete (or rename) the `STOP` file first — if it is still there, the trainer will save and exit again immediately — then rerun the **same** command with the same `--out` and `--resume`. `--resume` loads `runs/v2-myspecialist/latest.pt` when that file exists.

```bash
# Linux / macOS
rm -f runs/v2-myspecialist/STOP

# Windows PowerShell
Remove-Item -ErrorAction SilentlyContinue runs\v2-myspecialist\STOP
```

Then launch the same `python v2/seed_v2.py ... --out runs/v2-myspecialist ... --resume` command as before.

### 2.6 What to share if you want to contribute a v2.0 specialist

There is no v2.0 equivalent of `scripts/export_seed.py` yet (that script loads Seed v4 checkpoints only). To contribute a specialist, share:

1. **`best_ce.pt`** — the best-validation-CE snapshot (EMA weights + metrics, no optimizer). That is the file other people should load. Include `best_bpb.pt` as well if you like. `latest.pt` is much larger (it includes the optimizer) and is only needed if someone wants to resume *your* training run.
2. Confirmation you trained with **`tokenizers/seed-bpe150k-v2/tokenizer.json`**, SHA-256 `fc67f00b300e872734502b96eeee00440be6266bfe83338a164acfb7ca847d73`.
3. The shared socket dimensions: **`--d 512 --e-dim 128`**, plus the final shape (`layers` × `organs` / cells) and parameter count from `metrics.csv` or `train.log`.
4. Final mixed **`val_ce`** and **`val_bpb`** (and per-source numbers if you have them).
5. A short description of the skill, and a **license statement for the training data** (permissive / commercial-safe only — name the licenses). Do not upload the copyrighted or non-permissive source text.

---

## 3. Option B: Grow a Pool 5 (v1 4k Tokenizer) Seed & Wire It Into the Looped Top-6 Router

Use this path when you want a specialist that plugs into the published **Pool 5** organs and the winning **confidence-gated looped Top-6** router.

**Use the official 4,096-token tokenizer and socket**, or the new seed will not load with the published pool:

- Tokenizer: `tokenizers/seed-bpe4096-v1/tokenizer.json` (id `seed-bpe4096-v1`; SHA-256 `42ca2e21f41c629d2efb5c329deb64c4c168d80f3e3903c218342b589b6b2afa`; the manifest records the 16-character prefix `42ca2e21f41c629d`)
- Socket: `socket/seed-socket-v1`

Do **not** run `prep_data.py` without `--tokenizer`. Training a fresh BPE produces a seed that is not compatible with `seed-socket-v1` or the published organs.

### 3.1 Prepare data (`prep_data.py`)

`prep_data.py` writes `data/train.bin`, `data/valid.bin`, and `data/meta.json` (uint16 token files) using the official 4k tokenizer.

From a folder of your own `.txt`, `.md`, or `.pdf` files (PDFs need `pypdf`):

```bash
python prep_data.py --data-dir path/to/your_texts --out data \
    --tokenizer tokenizers/seed-bpe4096-v1/tokenizer.json
```

Or from TinyStories-style train/valid text files (documents separated by `<|endoftext|>`):

```bash
python prep_data.py --train-text TinyStoriesV2-GPT4-train.txt \
    --valid-text TinyStoriesV2-GPT4-valid.txt --out data \
    --tokenizer tokenizers/seed-bpe4096-v1/tokenizer.json
```

`--train-mb` (default 100), `--valid-mb` (default 4), and `--tok-mb` (default 25) cap how much of those text files is read when you use `--train-text` / `--valid-text`.

### 3.2 Train with `seed_v4.py`

`seed_v4.py` starts small (**2 layers × 1 cell** by default) and grows under a weights cap and a VRAM cap. The defaults fit a GTX 1660 Ti (6 GB). You can add a specialty corpus as a plain UTF-8 `.txt` with `--mix` (the flag can be repeated).

```bash
python seed_v4.py --out runs/myseed --device cuda --resume --data data \
    --tokenizer-dir tokenizers/seed-bpe4096-v1 \
    --socket-dir socket/seed-socket-v1 \
    --vram-cap-mb 5000 --mem-cap-mb 2048 \
    --minutes 1440 --mix path/to/specialty.txt
```

Windows helper with the same idea:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_seed_v4.ps1 -Out runs\myseed -Minutes 1440 -Device cuda -Data data -Mix path\to\specialty.txt
```

CPU instead of GPU:

```bash
python seed_v4.py --out runs/myseed --device cpu --threads 3 --minutes 180 --data data \
    --tokenizer-dir tokenizers/seed-bpe4096-v1 --socket-dir socket/seed-socket-v1
```

Useful v4 defaults to know about: `--batch 16 --accum 4 --ctx 512 --max-layers 24 --max-cells 6 --max-params 25000000` (25M), `--vram-cap-mb 5000`, `--mem-cap-mb 2048`. `--minutes` is the training horizon used by the learning-rate schedule (the command above uses one day; the code default is `20160` minutes if you omit it).

Monitor the run in `--out` (for example `runs/myseed/`):

| file | what it is |
|---|---|
| `train.log` | human-readable log, including `SAMPLE` generations (v4 writes samples to the log, not to a `samples.txt`) |
| `metrics.csv` | columns: `step, minutes, train_loss, val_ts, val_mix, val_ts_ema, val_mix_ema, layers, organs, params, lr_mult, phase, decor, balance, tok_s, step_ms, data_wait_ms, ckpt_layers, hunger, rss_mb, vram_mb` |
| `events.jsonl` | growth and other trainer events |
| `latest.pt` | resume checkpoint |

Pause with a `STOP` file in the run directory (`runs/myseed/STOP`); the trainer saves and exits. **Remove the `STOP` file**, then resume with the same `--out` and `--resume`.

### 3.3 Export an fp16 `.safetensors` organ (`scripts/export_seed.py`)

This drops optimizer state and writes a shareable **fp16** organ: `organs/seeds/myseed.safetensors` plus a `organs/seeds/myseed.json` manifest (typically 8–20 MB). This is what the pool and `chat.py` expect.

```bash
python scripts/export_seed.py runs/myseed/latest.pt organs/seeds/myseed \
    --name myseed --description "what this seed learned"
```

Optional flags: `--name` (defaults to the dest filename), `--description` (default: `Whole frozen seed, EMA weights, fp16.`), `--device` (default `cpu`).

### 3.4 Register it (`scripts/pool_register.py`)

Paths in the registry are relative to the repo root.

```bash
python scripts/pool_register.py --dir pool5 list
python scripts/pool_register.py --dir pool5 add myseed organs/seeds/myseed.json --desc "what it learned"
```

The `seed_pool.py` equivalent also works:

```bash
python seed_pool.py --dir pool5 add myseed organs/seeds/myseed.json --desc "what it learned"
```

To drop a seed later: `python scripts/pool_register.py --dir pool5 remove myseed`.

`add` / `remove` change the router key set, so you must retrain the looping router next. To rebuild a registry of the 16 published Pool 5 organs: `python scripts/pool_register.py --dir pool5 init`.

### 3.5 Train / retrain the confidence-gated looped Top-6 router (`pool5/eval/pool5_loop.py`)

Pool 5's published winner is **`--mode single --init-k 6 --step-k 3`** (a single-head router that activates 6 seeds on pass 1, then can add 3 more on a low-confidence pass). After any add or remove, run `base` then `loop`.

Copy the example mix and point every `"path"` at local UTF-8 `.txt` files (missing files are skipped; if none remain, training falls back to `data/train.bin`):

```bash
# Linux / macOS
cp pool5/mix.example.json pool5/mix.json
# Windows: copy pool5\mix.example.json pool5\mix.json
# then edit pool5/mix.json
```

Each row is `{ "name", "path", "weight" }`. Then:

```bash
python pool5/eval/pool5_loop.py base --dir pool5 --mode single --init-k 6 --step-k 3 --mix-json pool5/mix.json
python pool5/eval/pool5_loop.py loop --dir pool5 --minutes 15 --mix-json pool5/mix.json
```

- `base` trains the sparse context router (the seeds stay frozen) and writes `pool5/router_base.pt`. Default `--max-steps` is 3000, `--batch` 12, `--T` 384.
- `loop` trains the 1–3 pass confidence head on top (the base router stays frozen) and writes `pool5/router_loop.pt`. Default `--minutes` is 15.

`python pool5/eval/pool5_loop.py list --dir pool5` prints the registered seeds and whether `router_base.pt` / `router_loop.pt` are present.

The published winner weights already in the repo are `pool5/router_base.pt` and `pool5/router_loop.pt`. Other variants trained on the same 16 seeds live under `pool5/variants/`.

There is also a dense (all-active) router in `seed_pool.py`, which is a different architecture than the looped Top-6 winner:

```bash
python seed_pool.py --dir pool5 train --max-steps 900
python seed_pool.py --dir pool5 calibrate myseed --steps 200
```

Prefer `pool5_loop.py` when you want the published Pool 5 routing behavior.

### 3.6 Chat with the pool and tools

Looped Top-6 (greedy by default, confidence threshold `--tau 0.9`):

```bash
python pool5/eval/pool5_loop.py chat --dir pool5 --tools --prompt "What is 17 * 19?"
```

Omit `--prompt` for an interactive prompt (`/quit` or `/q` to exit). `--temp` default `0.0` is greedy; `--n` (default 200) is the generation length.

The same pool can also be chatted through `seed_pool.py`:

```bash
python seed_pool.py --dir pool5 chat --tools --prompt "What is 17 * 19?"
```

`--tools` uses the runtimes in [`tools/`](tools/README.md): `calculator`, `read_file`, `python_exec`, and optional `wiki_search` (set `SEED_WIKI` to a local introductions index; without it, `wiki_search` returns a short error). You can also chat with a single exported organ:

```bash
python chat.py organs/seeds/myseed.json
python chat.py organs/seeds/myseed.json --prompt "Once upon a time" --n 120
```

---

## 4. Compatibility Rules for Sharing Seeds

Follow these if you want other people to be able to load your seed, and if you want it considered for a community pool.

### Seed v2.0 specialists (Option A)

- **Tokenizer:** exactly [`tokenizers/seed-bpe150k-v2/tokenizer.json`](tokenizers/seed-bpe150k-v2/tokenizer.json). SHA-256 must be `fc67f00b300e872734502b96eeee00440be6266bfe83338a164acfb7ca847d73`. Do not train a replacement 150k BPE.
- **Socket dimensions:** `--d 512` and `--e-dim 128` (factorized tied embedding, `150000 × 128`). Other widths will not line up with everyone else's seeds.
- **Dynamic growth:** use `v2/seed_v2.py` as-is (`--start-layers 4 --start-cells 2`, growing up to `--max-layers 24 --max-cells 6` unless you have a reason to cap earlier). Do not invent a parallel architecture.
- **Data:** permissive / commercial-safe sources only (public domain, CC BY / CC BY-SA with attribution, ODC-BY, Apache-2.0, MIT, BSD, CDLA-Sharing, and similar). Do not train a shareable seed on scraped books, “all rights reserved” text, or other non-permissive material. Write the licenses into your `mix.json` `license` fields and into whatever note you send with the seed.
- **What to ship:** `best_ce.pt` (see [§2.6](#26-what-to-share-if-you-want-to-contribute-a-v20-specialist)), plus tokenizer SHA, socket flags, final shape / param count, `val_ce` / `val_bpb`, and the data-license note. There is not yet a v2.0 fp16 `.safetensors` exporter.

v2.0 seeds **cannot** be dropped into Pool 5.

### Pool 5 / Seed v4 organs (Option B)

- **Tokenizer:** exactly `tokenizers/seed-bpe4096-v1/tokenizer.json` (pass it to `prep_data.py --tokenizer` and `seed_v4.py --tokenizer-dir tokenizers/seed-bpe4096-v1`). SHA-256 `42ca2e21f41c629d2efb5c329deb64c4c168d80f3e3903c218342b589b6b2afa`.
- **Socket:** `socket/seed-socket-v1` (`--socket-dir socket/seed-socket-v1`). Organs are not compatible across different anchors or tokenizer versions.
- **Dynamic growth:** use `seed_v4.py` (default start 2×1, `--max-layers 24 --max-cells 6`).
- **Data:** the same permissive / commercial-safe rule as above.
- **Export:** `python scripts/export_seed.py runs/myseed/latest.pt organs/seeds/myseed --name myseed --description "..."`. This writes an fp16 `.safetensors` file plus a JSON manifest (`kind: tree`, `format: seed-part/1`, EMA weights, tokenizer and socket metadata, SHA-256, param count, shape). That pair is what you register and share. Then add it with `scripts/pool_register.py` and retrain the looped Top-6 router.

Starter organs in this repository are free. Anyone can train and share their own organs under the project's Apache-2.0 license.

---

## Quick reference

```bash
# Option A — Seed v2.0 (150k tokenizer, recommended)
python v2/seed_v2.py --tokenizer tokenizers/seed-bpe150k-v2/tokenizer.json \
    --mix-json v2/mix.json --out runs/v2-myspecialist --device cuda --resume \
    --d 512 --e-dim 128 \
    --start-layers 4 --start-cells 2 --max-layers 24 --max-cells 6 \
    --batch 8 --accum 4 --vram-cap-mb 4800 --mem-cap-mb 2048 \
    --max-params 150000000 --target-ce 1.0

# Option B — Pool 5 compatible (4k tokenizer)
python prep_data.py --data-dir path/to/your_texts --out data \
    --tokenizer tokenizers/seed-bpe4096-v1/tokenizer.json
python seed_v4.py --out runs/myseed --device cuda --resume --data data \
    --tokenizer-dir tokenizers/seed-bpe4096-v1 --socket-dir socket/seed-socket-v1 \
    --vram-cap-mb 5000 --mem-cap-mb 2048 --minutes 1440 --mix path/to/specialty.txt
python scripts/export_seed.py runs/myseed/latest.pt organs/seeds/myseed \
    --name myseed --description "what this seed learned"
python scripts/pool_register.py --dir pool5 add myseed organs/seeds/myseed.json --desc "what it learned"
python pool5/eval/pool5_loop.py base --dir pool5 --mode single --init-k 6 --step-k 3 --mix-json pool5/mix.json
python pool5/eval/pool5_loop.py loop --dir pool5 --minutes 15 --mix-json pool5/mix.json
python pool5/eval/pool5_loop.py chat --dir pool5 --tools --prompt "What is 17 * 19?"
```
