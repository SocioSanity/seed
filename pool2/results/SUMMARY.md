# pool2 results — specialist seeds (json1, answer1) + fair Qwen baseline
2026-11-24 (CST). One test run per model, greedy decoding everywhere. pool1 unchanged.

## What was built
| part | details |
|---|---|
| json1 | Seed v4 from scratch (socket v1 anchor, seed-bpe4096-v1), 4x2, 4.43M params, ctx 1024. Mix: 45% synth_json (tool-call arg filling, JSON read/write/fix, 120k docs) + glaive/calc/tools_misc/wiki_tool. Plateau stop at step 4750 (val_mix_ema 0.6815; 20.56 at birth). seed_json1 (local run folder) (final.pt, ARCHIVED.txt) |
| answer1 | Seed v4 from scratch, 4x2, 4.43M params. Mix: 60% synth_answer (17,825 real wiki_search episodes -> first sentence of the matching article; 20 test entities excluded) + 40% wiki_tool. Plateau stop at step 4500 (val_mix_ema 0.9072). seed_answer1 (local run folder) |
| pool2 | copy of pool1 router/registry + hot-added json1, answer1 (4 seeds). Router retrained on pool1 mix + 12% synth_json + 12% synth_answer (first 99% only); flattened at step 900 (268 s). pool2/ |
Neither seed grew past its 4.43M birth shape before the plateau stop (inside the 4-8M range).

## Main table (greedy; facts = used wiki / correct query / answer ok)
| | Qwen2.5-Coder-3B Q4_K_M native tools* | Qwen closed-book | Qwen our format + 3 few-shot | 100M (ref, sampled, earlier run) | run4 | pool1 | pool2 |
|---|---|---|---|---|---|---|---|
| bpb wiki_tool (seed held-out) | 0.751** | | | 0.666 | 0.770 | 0.780 | **0.716** |
| bpb glaive | 0.384 | | | 0.636 | 0.734 | 0.741 | **0.670** |
| bpb instruct | not run | | | 1.123 | 1.215 | 1.227 | 1.206 |
| bpb calc | not run | | | 0.106 | **0.204** | 0.229 | 0.237 |
| bpb tools_misc | not run | | | 0.330 | 0.502 | 0.423 | **0.419** |
| bpb gsm8k | not run | | | 0.743 | **0.821** | 0.841 | 0.840 |
| bpb TinyStories | 0.576 | | | 0.547 | 1.031 | 0.533 | **0.533** |
| tool calls / valid JSON | 88 / 100% (0 tagged; 38 duplicates) | 0 | 0 | 22 / 100% | 38 / 100% | 38 / 100% | 39 / 100% |
| 20 facts | 20 / 20 / 20 | – / – / 20 | 0 / 0 / 0 | 17 / 13 / 12 | 17 / 17 / 11 | 17 / 17 / 11 | 19 / 19 / **16** |
| 15 held-out obscure facts | 15 / 15 / **12** | – / – / 2 | 0 / 0 / 0 | not run | 15 / 15 / 7 | 15 / 15 / 7 | 15 / 15 / **12** |
| 10 math | 8 (calc 10/10) | 8 | 0 | 2 | 5 | 5 | 5 |
| params | 3.09B | | | 104.5M | 3.72M | 8.55M | 17.4M (4 seeds + router) |
| weights MB | 1840.5 (Q4_K_M GGUF) | | | 398.8 fp32 | 14.2 fp32 | 32.6 fp32 | 66.4 fp32 / 33.2 fp16 |
| tok/s GPU / CPU(4 thr) | 85.6 / 6.3 (llama.cpp Vulkan / CPU) | | | 41 / 16 | 103 / 82 | 56 / 25 | 29 / 14 |
\* Native = Qwen chat template, calls re-rendered as `<tool_call>` and results sent as tool turns (`<tool_response>`), max 2 calls, then an explicit "now answer" user turn. 43/45 questions needed that forced step (the model re-issues the same call instead of answering); the seeds got no such help. Qwen never emits the `<tool_call>` tag; bare JSON calls were accepted leniently.
\** Qwen bpb: official HF Qwen/Qwen2.5-Coder-3B-Instruct weights (same model as the Ollama Q4_K_M blob, unquantized) run in **fp32** (GPU + CPU offload, isolated venv a separate venv, transformers 5.24.1), true log-probs, same docs and same bytes as the seeds (byte counts verified equal), bits / UTF-8 bytes, each doc prefixed with <|endoftext|>. fp16 gave 0.7506 vs fp32 0.7507 on wiki_tool. 100m_val versions: wiki_tool 0.772, glaive 0.378. Other sets not run (scope narrowed to wiki_tool + TinyStories). The GGUF Q4_K_M used for the functional tests was not scored for bpb. Exact prompts: results/qwen_prompts.md.
Answer checks are keyword-based (lenient): a garbled final that contains the key word still counts (see samples).

Qwen bpb note: wiki_tool docs are in the seeds' own tool-transcript format (`<|system|>`, `<tool_call>` tags), which favours seeds trained on that format; glaive (generic tool-chat) and TinyStories are fairer comparisons. Qwen beats pool2 on glaive (0.384 vs 0.670) but pool2 wins on wiki_tool (0.716 vs 0.751) and TinyStories (0.533 vs 0.576).

## Router behaviour (pool2 mean weight by text region; pool1 for comparison)
| task / region | run3 | run4 | json1 | answer1 | pool1 run4 |
|---|---|---|---|---|---|
| fact think | 0.00 | 0.56 | 0.33 | 0.11 | 1.00 |
| fact tool_call | 0.00 | 0.28 | **0.61** | 0.11 | 1.00 |
| fact tool_result | 0.00 | 0.15 | 0.10 | **0.75** | 0.99 |
| fact answer | 0.00 | 0.24 | 0.40 | 0.36 | 1.00 |
| obscure tool_call | 0.00 | 0.29 | **0.59** | 0.12 | 1.00 |
| obscure tool_result | 0.00 | 0.15 | 0.08 | **0.77** | 0.99 |
| obscure answer | 0.00 | 0.29 | 0.30 | **0.41** | 1.00 |
| math think | 0.00 | **0.72** | 0.25 | 0.03 | 1.00 |
| math tool_call | 0.00 | 0.26 | **0.70** | 0.04 | 1.00 |
| story | **0.90** | 0.09 | 0.01 | 0.00 | 0.10 |
The router uses the specialists where they were meant to go: json1 on tool calls, answer1 on reading tool results and answers, run4 on reasoning, run3 on stories.

## Samples
- pool2, obscure "What is German submarine U-454?" -> searched "German submarine U-454" -> "German submarine U-454 was a Type VIIC U-boat of Nazi Germany's Kriegsmarine during World War II." (correct; run4/pool1 missed it)
- pool2, "Who was Charles Darwin?" -> "Charles Robert Darwin ( DAR-win; 12 February 1809 – 19 April 1882) was an English naturalist, geologist, and b..."
- pool2 failures: "Who were the Beatles?" -> "The Beatles were the Beatles."; Everest answer starts right then degenerates ("SagarmŁth_Nepal border ... COP 27- 20222"); "Gashtiari (, also Romanized as Gasht}}}}" (brace glitch); Einstein answer counts as correct but begins "Albert Einstein House at 112 March 1879 ..." (mixes two results).
- pool2 math: arithmetic calls perfect (5/5), all 5 word problems wrong, same as run4.
- pool2 stories: plain story fluent (run3 carries it); chat-format story is one line ("A dog who is afraid of the dark.").
- Qwen native, obscure: correct calls every time; answers fluent and mostly right; misses were a missing year (Bobby Anderson), a wrong date (Bahá'í Samoa "1954", lead says 1916), Gashtiari without the district.
- Qwen closed-book on obscure: confident hallucinations ("Andrew Kenny is a Canadian politician who served as the 18th Prime Minister of Canada") or refusals; 2/15.
- Qwen in our format with 3 few-shot examples: writes `<think>...</think>` then ends the turn every time (greedy) -> format sensitivity, 0 calls.

## Verdict (honest)
- Specialists helped, and only where intended: pool2 vs pool1 = facts answered 11 -> 16/20, obscure 7 -> 12/15, bpb wiki_tool -0.064, glaive -0.071, TinyStories unchanged, JSON still 100%. Math unchanged (5/10); calc bpb slightly worse.
- pool2 ties Qwen2.5-Coder-3B Q4_K_M on the held-out obscure facts (12/15 each) at ~1/28 the file size (66 MB fp32 vs 1840 MB Q4) — but Qwen needed a forced "now answer" step, and Qwen's answers are fluent and rewritten while pool2 mostly copies the lead's first sentence, sometimes garbled. On memorized facts and math Qwen wins (20/20 and 8/10 vs 16/20 and 5/10).
- Qwen is very format-sensitive: 0 calls in our format even with few-shot; tool use only via its native template, and it loops without the forced answer step.
- Costs: pool2 runs all 4 seeds per token -> 29 tok/s GPU / 14 CPU (pool1 56/25, run4 103/82). Qwen 3B: 86 GPU / 6 CPU.
- pool1 and run4 give identical greedy outputs (router ~1.0 on run4 for tool text), so pool1 added nothing on tool tasks; pool2 is the first pool that beats its best single seed.

## Files
results/summary.csv, bpb.csv, functional.csv, models.csv, router_regions.csv, transcripts.jsonl (seed models, greedy, incl. obscure);
results/qwen_fair_{native,closed,fewshot}_*.csv/jsonl, qwen_speed.csv, qwen_bpb_SKIPPED.txt, qwen_prompts.md;
results/qwen_functional.csv / qwen_native_* = earlier sampled (temp 0.7) Qwen runs (no few-shot / no forced answer), kept for reference; archive_* = superseded Qwen runs.
heldout_obscure_facts.json (15 entities: titles absent from every training file; answer keys hand-picked from the lead); pipeline.py/log; test_pool2.py (greedy + obscure; pre-change copy test_pool2_pre_greedy.py); test_qwen3b_fair.py.
