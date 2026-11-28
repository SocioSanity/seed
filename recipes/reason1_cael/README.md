# reason1_cael recipe

Helpers used to prepare and evaluate the **UNDERTRAINED / experimental** Seed v2.0 organ [`reason1-cael-undertrained`](../../organs/seeds/reason1-cael-undertrained.md).

The CAEL parquet files, GSM8K, SmolTalk, and FineWeb-Edu text are **not** in this repo. Point the scripts at your local copies.

```bash
# tokenizer (optional; the published 8192 BPE is already in tokenizers/seed-bpe8192-r1/)
python recipes/reason1_cael/build_tokenizer.py --out tokenizers/seed-bpe8192-r1

# format CAEL + GSM8K into .txt sources and held-out jsonl
python recipes/reason1_cael/prep_data.py --cael-raw path/to/cael/parquet --out-text path/to/text

# copy mix.example.json and edit the paths, then grow (CPU example):
python v2/seed_v2.py --out runs/reason1 --mix-json v2/reason1_cael.mix.json \
    --tokenizer tokenizers/seed-bpe8192-r1/tokenizer.json --device cpu --threads 4 \
    --priority below_normal --ctx 512 --batch 4 --accum 2 --e-dim 128 --d 512 \
    --start-layers 4 --start-cells 2 --target-ce 0.01 --resume

# exact-match eval (copies the checkpoint first; does not touch a live trainer)
python recipes/reason1_cael/eval_r1.py runs/reason1/best_ce.pt tag --n-cat 30 --n-hard 15 --n-gsm 100
```

See `train_args.txt` for the flags used on the published run, and [DATA_SOURCES.md](../../DATA_SOURCES.md) for licenses and the CAEL citation.
