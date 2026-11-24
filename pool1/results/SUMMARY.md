# Seed Pool v1 (pool1): run3 + run4 as organs behind a trained router (2026-11-24, ~12:00 CST, one test run)

**Verdict:** pool1 does what the failed merge could not. It keeps run4's tool and reasoning skill (bpb within 1–3% of run4; same query and math scores, within run-to-run noise) and run3's story fluency with no loss (TinyStories 0.533 bpb vs run3's 0.536 — a tie, not an improvement). Against the 100M it is in the same league on this early checkpoint: 16/20 correct queries vs 13, 9/20 right answers vs 12, valid JSON 91% vs 100%, math 4/10 vs 2. It is still ~0.1 bpb behind on tool text. It is 32.6 MB vs 399 MB (about 12× smaller) and 1.7× faster on GPU / 2.6× on CPU. That 100M is only at step 1,350 of 3,662 planned, so it is undertrained; “close to the 100M” is a comparison against this early baseline. Future comparisons will use Qwen 3B coder. Short final answers (summarising the tool result) are the weakest part; that is run4's limit, not the pool's.

## Design (seed_pool.py)

In this document an **organ** is a whole frozen seed in the pool. (The older per-column `seed-organ/1.1` bundles are **organelles**; see the failed-merge section.)

- Each active organ runs on every token; prediction = mixture of the organs' distributions: log p = logsumexp_i(log g_i + log_softmax(logits_i)).
- Router (403k params, 1.5 MB): GRU over a 64-d token embedding (context) + each organ's own final normalized hidden state + its entropy/max log-prob + a learned per-organ key -> score per organ -> softmax over the active organs. Works for any number of organs; a new organ gets a key (initialised to the mean of the others) and optionally a key-only calibration.
- Registry (pool1/registry.json), 2 GB weights budget, hot add/remove, usefulness = running average router weight (evicts the least useful organ when over budget), prefetch to CPU. Seeds longer than their own context use sliding windows (run3 512, run4 1024; the pool window is 1024). Brace decode prior applied when sampling.
- Router training: router only, organs frozen, TinyStories train.bin (40%) + run4 tool/reasoning files' first 99% (60%), never the held-out tail. 800 steps, 2 minutes, stopped when val flattened (val TinyStories 1.776 -> 1.530, tool 1.345 -> 0.869 nats/token).

## Results (same harness, sets and seeds as the merged1 test; the 100M's numbers reused from that run)

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

Bits per byte are on the seed held-out set (last 1% of run4's files; lower is better). The 100M's own held-out set is in bpb.csv (same picture). The 20-item facts and 10-item math rows are small and noisy (about ±1 run to run). Sampling on the GPU is not bit-for-bit repeatable: run4 scored 12 answers / 5 math in the earlier merged1 run and 11 / 4 here. Neither code base has a KV cache.

On facts, the pool had 16 correct queries and 9 right answers (run4: 16 and 11; the 100M: 13 and 12). Stories: pool 0.533 vs run3 0.536 is a tie (no loss), not a win.

## Router behaviour (pool1, mean router weight on run4 by text region, functional suite)

| task | region | tokens | weight on run4 |
|---|---|---|---|
| fact | think | 393 | 0.9996 |
| fact | tool_call | 922 | 0.9999 |
| fact | tool_result | 6,824 | 0.991 |
| fact | answer | 1,737 | 0.998 |
| math | think / tool_call / answer | 990 | 0.999–1.0 |
| story | story | 203 | 0.138 (10% of tokens > 0.5) |

On training val: TinyStories → run3 0.97; tool text → run4 0.97. The router switches per token, e.g. a story prompt inside the chat format starts on run4 and moves to run3.

## Failed merge (merged1)

Transplanting run4 into the run3 trunk with `parts_tool.py import` (harvest run4's only column as a `seed-organ/1.1` organelle bundle, then adopt) is **experimental** and is **known to fail for whole-seed merges**. merged1 kept run3's story fluency but lost run4's tool skill. Prefer the Seed Pool, which keeps whole frozen seeds as organs and mixes them with a router.

These numbers are from the earlier merged1 run of the same harness (2026-11-24, ~10:45 CST). run4's functional scores in that run were 12/20 answers and 5/10 math (vs 11 and 4 in the pool1 run above); treat ±1 as noise. tiny100m is the same undertrained step-1,350 baseline.

| | run3 | run4 | merged1 | tiny100m (step 1,350) |
|---|---|---|---|---|
| params | 4.43M | 3.72M | 8.00M | 104.5M |
| weights MB fp32 / fp16 | 16.9 / 8.5 | 14.2 / 7.1 | 30.5 / 15.3 | 398.8 / 199.4 |
| tok/s GPU / CPU (4 threads) | 166 / 138 | 130 / 128 | 56 / 57 | 41 / 16 |
| bpb wiki_tool | 4.489 | 0.770 | 2.885 | 0.666 |
| bpb glaive | 4.082 | 0.734 | 2.801 | 0.636 |
| bpb instruct | 3.543 | 1.215 | 2.912 | 1.123 |
| bpb calc | 5.352 | 0.204 | 2.796 | 0.106 |
| bpb tools_misc | 3.511 | 0.502 | 2.099 | 0.330 |
| bpb gsm8k | 3.920 | 0.821 | 2.492 | 0.743 |
| bpb TinyStories | 0.536 | 1.031 | 0.537 | 0.547 |
| tool calls / valid JSON | 0 / - | 22 / 100% | 22 / 59% | 22 / 100% |
| facts: correct queries / right answers (of 20) | 0 / 0 | 16 / 12 | 0 / 1 | 13 / 12 |
| math ok (of 10) | 0 | 5 | 1 | 2 |
| fluency by eye | good | poor | good | good |

merged1 also used wiki on 8 of 20 fact items and the calculator on 6 of 10 math items, but queries were garbled (for example “Alber Cerrricumistal”), so searches missed. Interference from the host's always-on cells is the likely cause: run4's organelle was the whole of run4's network, trained on its own residual stream.

## Hot add / remove demo (on a copy; see hotswap_demo.txt)

- Added run2 (older socket seed, never seen by the router) in 0.3 s: weights moved on stories (run3 0.49 / run2 0.40), tool text stayed on run4 (0.994).
- Key-only calibration, 100 steps / 21 s: run2 settled to 0.16 on stories, 0.001 on tools; val unchanged (1.133).
- Removed run2: the pool's log-probs are bit-identical to before the add (max diff 0.0).
- Budget 32 MB + re-add run2: the least-used organ (run4, after a story-heavy stretch) was evicted automatically; prefetched run4 was re-added in 0.01 s.
- Caveat: usefulness is recent usage, so a story-heavy stretch can evict the tool organ; task-aware eviction or keeping an organ pinned would fix that.

## Files

seed_pool.py, router.pt + registry.json + train.log + metrics.csv in pool1/; results/: bpb.csv, models.csv, summary.csv, router_regions.csv, hotswap_demo.txt. This file: [pool1/results/SUMMARY.md](https://github.com/SocioSanity/seed/blob/main/pool1/results/SUMMARY.md).

CLI: `python seed_pool.py list | chat --tools --prompt "Who was Marie Curie?" | add NAME PATH | remove NAME | calibrate NAME | train`.
