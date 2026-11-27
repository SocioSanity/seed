# Seed Pool 5 results

16 frozen specialist organs on `seed-bpe4096-v1` and `seed-socket-v1`. Total **88.7M parameters**, **338.5 MB** fp32 weights (about 169 MB published fp16). Winner: **single-head Top-6** looping router (`--mode single --init-k 6 --step-k 3`, confidence `tau=0.9`).

## Router variants (same 16 seeds, greedy)

| variant | macro BPB | greedy tok/s (GPU) | coding eval (20 Python tasks) |
|---|---|---|---|
| **single-head Top-6 (winner)** | **0.8293** | **17.2** | 0/20 chat, 0/20 complete |
| single-head Top-3 | 0.8375 | 26.7 | 0/20 |
| multi-head Top-3 (3 heads) | 0.8959 | 19.4 | 0/20 |

## Top-6 domain bits per byte (held-out, lower is better)

| domain | BPB | mean passes | docs |
|---|---|---|---|
| python | 0.4472 | 1.003 | 24 |
| C | 0.6490 | 1.131 | 24 |
| JavaScript | 1.0417 | 1.166 | 24 |
| bash | 1.1178 | 1.021 | 24 |
| PowerShell | 0.8088 | 1.090 | 24 |
| alg (real) | 1.6281 | 1.380 | 24 |
| alg (synth) | 1.3755 | 1.279 | 16 |
| math | 0.1740 | 1.073 | 16 |
| reason | 0.0750 | 1.058 | 16 |
| Colby texts | 0.9759 | 1.074 | 22 |
| **macro** | **0.8293** | | |

What worked: the pool is usable in real time on a GTX 1660 Ti, English is fluent, and domain BPB is strong on math, reasoning, and the code/shell languages the specialists were trained on.

What we diagnosed honestly: on a strict 20-task Python coding eval, both the Top-6 pool and the `py1` organ alone scored **0/20**. The 4,096-token BPE, the ~4–10M per-seed size cap, and router prose bleed (non-Python organs still getting weight on coding prompts) are the bottleneck we are targeting in Seed v2.0 — not a claim that the architecture failed.

## How to run

```bash
python pool5/eval/pool5_loop.py chat --dir pool5 --tools --prompt "What is 17 * 19?"
python scripts/pool_register.py --dir pool5 list
```

To drop in a new seed and retrain the same looping Top-6 architecture, see the root README section "Link seeds and route them with looped Top-6".

Raw numbers from the original run are in this folder (`top6_bpb.json`, `top6_code.json`, `diag_codetest.json`, and the Top-3 / multi-head counterparts). Local absolute paths in those files, if any, refer to the training machine and are not required to load the published organs.
