# Seed Pool 5

16 frozen organs and the winning **single-head looping Top-6** router.

```bash
python scripts/pool_register.py --dir pool5 list
python pool5/eval/pool5_loop.py chat --dir pool5 --tools --prompt "What is 17 * 19?"
```

- `registry.json` — seed names, relative organ paths, sizes
- `router_base.pt` / `router_loop.pt` — published Top-6 winner
- `variants/top3`, `variants/mh3` — the other two routers trained on the same 16 seeds
- `eval/pool5_loop.py` — train `base` / `loop` and `chat`
- `eval/pool5_eval.py` — held-out BPB / coding eval (needs your data files)
- `mix.example.json` — template for retraining the router on your own text
- `results/` — numbers from the original run; see [results/SUMMARY.md](results/SUMMARY.md)

After you add or remove a seed, retrain:

```bash
python pool5/eval/pool5_loop.py base --dir pool5 --mode single --init-k 6 --step-k 3 --mix-json pool5/mix.json
python pool5/eval/pool5_loop.py loop --dir pool5 --minutes 15 --mix-json pool5/mix.json
```
