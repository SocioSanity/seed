# Seed Pool v1 (pool1): run3 + run4 as whole frozen seeds behind a trained router (2026-11-24, ~12:00 CST, one test run)

**Verdict:** pool1 does what the failed merge could not. It keeps run4's tool and reasoning skill (bpb within 1-3% of run4, same query and math scores) and run3's story fluency (TinyStories 0.533 bpb, slightly better than run3 alone). Against the 100M it is in the same league on tool use: 16/20 correct queries vs 13, valid JSON 91% vs 100%, answers 9/20 vs 12, math 4/10 vs 2. It is still ~0.1 bpb behind on tool text. It is 32.6 MB vs 399 MB (12x smaller) and 1.7x faster on GPU / 2.6x on CPU. Short final answers (summarising the tool result) are the weakest part; that is run4's limit, not the pool's.

## Design (seed_pool.py)
- Each active seed runs on every token; prediction = mixture of the seeds' distributions: log p = logsumexp_i(log g_i + log_softmax(logits_i)).
- Router (403k params, 1.5 MB): GRU over a 64-d token embedding (context) + each seed's own final normalized hidden state + its entropy/max log-prob + a learned per-seed key -> score per seed -> softmax over the active seeds. Works for any number of seeds; a new seed gets a key (initialised to the mean of the others) and optionally a key-only calibration.
- Registry (pool1/registry.json), 2 GB weights budget, hot add/remove, usefulness = running average router weight (evicts the least useful seed when over budget), prefetch to CPU. Seeds longer than their own context use sliding windows (run3 512, run4 1024; the pool window is 1024). Brace decode prior applied when sampling.
- Router training: router only, seeds frozen, TinyStories train.bin (40%) + run4 tool/reasoning files' first 99% (60%), never the held-out tail. 800 steps, 2 minutes, stopped when val flattened (val TinyStories 1.776 -> 1.530, tool 1.345 -> 0.869 nats/token).

## Results (same harness, sets and seeds as the merged1 test; the 100M's numbers reused from that run)
| | run3 | run4 | **pool1** | tiny100m |
|---|---|---|---|---|
| params | 4.43M | 3.72M | 8.55M | 104.5M |
| weights MB fp32 / fp16 | 16.9 / 8.5 | 14.2 / 7.1 | **32.6 / 16.3** | 398.8 / 199.4 |
| tok/s GPU / CPU (4 threads) | 165 / 138 | 133 / 135 | 68 / 42 | 41 / 16 |
| bpb wiki_tool | 4.489 | 0.770 | 0.780 | **0.666** |
| bpb glaive | 4.082 | 0.734 | 0.741 | **0.636** |
| bpb instruct | 3.543 | 1.215 | 1.227 | **1.123** |
| bpb calc | 5.352 | 0.204 | 0.229 | **0.106** |
| bpb tools_misc | 3.511 | 0.502 | 0.423 | **0.330** |
| bpb gsm8k | 3.920 | 0.821 | 0.841 | **0.743** |
| bpb TinyStories | 0.536 | 1.031 | **0.533** | 0.547 |
| tool calls / valid JSON | 0 / - | 23 / 100% | 23 / 91% | 22 / 100% |
| facts: used wiki / correct query / answer ok (of 20) | 0/0/0 | 16/16/11 | 16/16/9 | 17/13/12 |
| math ok (of 10) / used calculator | 0 / 0 | 4 / 7 | 4 / 7 | 2 / 5 |
| fluency by eye | good | poor | good | good |
bpb on the seed held-out set (last 1% of run4's files); the 100M's own held-out set is in bpb.csv (same picture). Sampling on the GPU is not bit-for-bit repeatable: run4 scored 12 answers / 5 math in the earlier run and 11 / 4 here, so treat +-1 as noise. Neither code base has a KV cache.

## Router behaviour (pool1, mean router weight on run4 by text region, functional suite)
| task | region | tokens | weight on run4 |
|---|---|---|---|
| fact | think | 393 | 0.9996 |
| fact | tool_call | 922 | 0.9999 |
| fact | tool_result | 6,824 | 0.991 |
| fact | answer | 1,737 | 0.998 |
| math | think / tool_call / answer | 990 | 0.999-1.0 |
| story | story | 203 | 0.138 (10% of tokens > 0.5) |
On training val: TinyStories -> run3 0.97; tool text -> run4 0.97. The router switches per token, e.g. a story prompt inside the chat format starts on run4 and moves to run3.

## Hot add / remove demo (on a copy; see hotswap_demo.txt)
- Added run2 (older socket seed, never seen by the router) in 0.3 s: weights moved on stories (run3 0.49 / run2 0.40), tool text stayed on run4 (0.994).
- Key-only calibration, 100 steps / 21 s: run2 settled to 0.16 on stories, 0.001 on tools; val unchanged (1.133).
- Removed run2: the pool's log-probs are bit-identical to before the add (max diff 0.0).
- Budget 32 MB + re-add run2: the least-used seed (run4, after a story-heavy stretch) was evicted automatically; prefetched run4 was re-added in 0.01 s.
- Caveat: usefulness is recent usage, so a story-heavy stretch can evict the tool seed; task-aware eviction or keeping a seed pinned would fix that.

## Files
seed_pool.py, router.pt + registry.json + train.log + metrics.csv in pool1/; results/: bpb.csv, models.csv, summary.csv, router_regions.csv, hotswap_demo.txt.
CLI: `python seed_pool.py list | chat --tools --prompt "Who was Marie Curie?" | add NAME PATH | remove NAME | calibrate NAME | train`.
