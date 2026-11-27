# v0.4.0 — Seed Pool 5, looping Top-6 router, and Seed v2.0 trainer

## Seed Pool 5

Sixteen frozen specialist organs (~88.7M parameters, 338.5 MB fp32 / ~169 MB fp16) on the shared 4k tokenizer and socket, plus a confidence-gated 1–3 pass looping router. The winner is the **single-head Top-6** router: **0.8293 macro BPB**, **17.2 tok/s** greedy on a GTX 1660 Ti.

Organs (fp16 tree parts in `organs/seeds/` and in `seed-pool-v5-organs.zip`): run3, run4, json1, answer1, math1, reason1, plan1, plan2, py1, fix1, inst1, c1, js1, sh1, ps1, alg1.

What worked: fast enough for interactive use, fluent English, strong domain BPB on math, reasoning, and code/shell. What we diagnosed honestly: **0/20** on a strict 20-task Python coding eval (same for `py1` alone). The 4k BPE, the ~4–10M per-seed size cap, and router prose bleed are the next bottleneck.

```bash
python seed_pool.py --dir pool5 list
python pool5/eval/pool5_loop.py chat --dir pool5 --tools --prompt "What is 17 * 19?"
```

## Grow your own (v4 pool-compatible, or v2.0 150k)

- `seed_v4.py` is the full dynamic-growth trainer, including the GPU optimizer-device fix (`_sync_opt_devices`). Use the official 4k tokenizer and socket so a new seed can join Pool 5.
- `scripts/export_seed.py` and `scripts/pool_register.py` export a seed and add it to a `registry.json`.
- `pool5/eval/pool5_loop.py` retrains and runs the same looped Top-6 router on whatever seeds are in the registry.
- `v2/seed_v2.py` plus `tokenizers/seed-bpe150k-v2/` (150k ByteLevel BPE, SHA-256 `fc67f00b…`) is the new line: factorized embeddings (`e_dim=128`, `d_model=512`), grows until val CE < 1.0. Not compatible with Pool 5.

Starter organs are free. Anyone can train and share their own organs.

If Seed is useful to you, donations help keep development going: Cash App [$ColbyMcgarrah0](https://cash.app/$ColbyMcgarrah0)
