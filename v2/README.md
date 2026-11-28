# Seed v2.0 trainer

Dynamic-growth trainer for the official 150k tokenizer. Factorized tied embeddings (`--e-dim 128`) and residual width `--d 512`. Starts at 4 layers × 2 cells and grows under a 2 GB weight cap. Stops when mixed validation cross-entropy is below `--target-ce` (default 1.0).

This is a **new** line. It does not use `seed-socket-v1` or the 4k tokenizer, and it is not drop-in compatible with Pool 5 organs.

Quick start is in the root [README](../README.md#grow-your-own-seed-v20-150k).

```bash
# Copy v2/mix.example.json and point every "path" at your own local .txt files.
python v2/seed_v2.py --tokenizer tokenizers/seed-bpe150k-v2/tokenizer.json \
    --mix-json v2/mix.json --out runs/v2-eng1 --device cuda --resume \
    --d 512 --e-dim 128 --target-ce 1.0 --vram-cap-mb 4800 --mem-cap-mb 2048
```

Create `runs/v2-eng1/STOP` to save and exit. Resume with the same `--out` and `--resume`.

## Experimental organ: reason1-cael-undertrained

An **UNDERTRAINED / experimental** CPU reasoning seed lives at [`organs/seeds/reason1-cael-undertrained`](../organs/seeds/reason1-cael-undertrained.md). It is Seed v2.0 (10.63M, 4x2) but uses the specialist tokenizer [`tokenizers/seed-bpe8192-r1/`](../tokenizers/seed-bpe8192-r1/), not the official 150k BPE. It is not drop-in compatible with Pool 5 or with a 150k v2 pool.

```bash
python v2/chat_v2.py organs/seeds/reason1-cael-undertrained.json
python scripts/export_seed_v2.py path/to/best_ce.pt organs/seeds/myseed --tokenizer-dir tokenizers/seed-bpe8192-r1
```
