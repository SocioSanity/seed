# tiny100m comparison model

This is a **comparison-only** 104.5M-parameter model (best EMA at step 1,350 of 3,662 planned, so it is undertrained). It is not a Seed, does not use `seed-socket-v1`, and is not the product. It is here so you can run the same kind of test Seed Pool v1 was measured against. Future comparisons will use Qwen 3B coder.

The weights are about 200 MB in fp16, so they are not in the git repo. Download them from the [v0.2.0 release](https://github.com/SocioSanity/seed/releases/tag/v0.2.0):

- `tiny100m-best-step1350-fp16.safetensors`
- `tiny100m-config.json` and `tiny100m-tokenizer.json` (also inside `tiny100m-compare.zip`)

Place the three files in this folder, or pass explicit paths:

```bash
python comparison/tiny100m/load_tiny100m.py --weights tiny100m-best-step1350-fp16.safetensors --prompt "Once upon a time"
```

`load_tiny100m.py` also loads a local `best.pt` training checkpoint if you have one. On load, fp16 tensors are cast back to fp32.

The architecture is a frozen ~12M TinyStories core (width 256, 14 layers, vocab 8,192) plus a 768-wide tower and an expanded 16,384-token vocabulary, context 512. See `model100m.py` and `core_model.py`. `chat100m.py` is a small optional chat/tool loop (calculator and `read_file`; set `SEED_WIKI` if you have a local Wikipedia leads index).

The comparison numbers used in the repo README and in `pool1/results/SUMMARY.md` came from this checkpoint at step 1,350.
