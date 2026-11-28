# reason1-cael-undertrained

**Status: UNDERTRAINED / experimental.** This is a first CPU reasoning seed, not a finished organ. It is published so the weights and recipe are on the record. Do not treat it as a reliable reasoner, and do not drop it into Pool 5.

It is also **not** the Pool 5 organ [`reason1-final`](reason1-final.json) (Seed v4, 4.43M, `seed-bpe4096-v1`). The two share a name family only.

## What it is

| | |
|---|---|
| Architecture | Seed v2.0 (`arch: seed-v2.0`), dynamically grown, never fixed-shape |
| Shape / size | **4×2**, **10.63M** parameters (factorized tied embeddings `8192 × 128` → `d=512`) |
| Context | 512 tokens |
| Precision | Trained CPU fp32; published weights are **fp16** EMA (same convention as other organs in this folder) |
| Tokenizer | [`tokenizers/seed-bpe8192-r1/`](../../tokenizers/seed-bpe8192-r1/) (8,192 ByteLevel BPE) |
| Format | `seed-v2-part/1` — `reason1-cael-undertrained.json` + `.safetensors` (20.3 MB) |
| Checkpoint | EMA weights from `best_ce.pt` at **step 17,000** (run stopped at step 17,011) |
| Lineage | `d46c50b8ed34` |
| Weights SHA-256 | `fce1b46c079492db8c2b0c578135b9497264a061ec0055c458916ab467e10ec8` |
| License (weights / code) | Apache-2.0 |

A deepen trial at step 16,800 (4×2 → deeper) was rejected (`keep: false`). The seed stayed 4×2 for the whole run.

Training: about **11 hours 25 minutes** on a desktop CPU (Intel i7-10700F, 4 torch threads, below-normal priority, so a GPU English-fluency run could keep going), **69.6M tokens**, ~1,910 tok/s. Mixed EMA validation at the published step: **val_ce 0.9743 / val_bpb 0.4516**. A slightly earlier best at step 15,750 was 0.9784 / 0.4536.

## Data mix

CAEL is the core (~82% sampling weight). A modest share of human-written and general text was added so the seed would not only see templated word problems.

| source | weight | license |
|---|---:|---|
| CAEL arithmetic | 0.18 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), [CAEL Verified Reasoning](https://huggingface.co/datasets/vynly/cael-verified-reasoning) data revision 1.1.1, Vovala14 (CAEL) |
| CAEL dates_times | 0.15 | same |
| CAEL composite | 0.14 | same |
| CAEL logic | 0.14 | same |
| CAEL conflicting_sources | 0.08 | same |
| CAEL memory | 0.06 | same |
| CAEL unanswerable | 0.02 | same |
| CAEL arc_grids (length-filtered) | 0.05 | same |
| GSM8K train | 0.04 | MIT ([openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k)) |
| SmolTalk | 0.08 | Apache-2.0 (HuggingFaceTB/smoltalk) |
| FineWeb-Edu | 0.06 | ODC-BY 1.0 (HuggingFaceFW/fineweb-edu) |

Rows were formatted as `User: …` / `Assistant: <worked solution>` / `Answer: <final>`. CAEL train was deduped on core question text (template wrappers stripped) and leakage-filtered against the official CAEL `test` and `hard` splits. Long ARC-style grids above 480 tokens were dropped. Official CAEL test/hard and GSM8K test were held out. Full attribution and BibTeX for CAEL are in [DATA_SOURCES.md](../../DATA_SOURCES.md).

## Held-out language-model CE (step 17,000, EMA)

Bits and nats per token on the trainer's mixed held-out set (CAEL official test, split by category; GSM8K test; SmolTalk / FineWeb-Edu val).

| source | val_ce (nats) | val_bpb |
|---|---:|---:|
| cael_arithmetic | 0.256 | 0.136 |
| cael_dates_times | 0.336 | 0.180 |
| cael_composite | 0.250 | 0.142 |
| cael_logic | 0.537 | 0.193 |
| cael_conflicting | 0.251 | 0.098 |
| cael_memory | 0.201 | 0.089 |
| cael_unanswerable | 0.247 | 0.094 |
| cael_arc_grids | 0.505 | 0.519 |
| gsm8k (test) | 2.415 | 1.137 |
| smoltalk | 3.873 | 1.429 |
| fineweb_edu | 5.059 | 2.108 |
| **mixed EMA** | **0.974** | **0.452** |

Low CE on templated CAEL categories is **not** evidence of real reasoning. GSM8K and general English stay much higher.

## Early exact-match eval (step 500 only)

This was run on the step-500 `best_ce.pt`, **not** on the published step-17,000 weights. Greedy decode, exact match of the `Answer:` line (the CAEL `verify.judge` helper agreed with exact match on this sample).

| split | n | exact match |
|---|---:|---:|
| CAEL official test (30 per category) | 240 | **17.1%** |
| CAEL hard | 119 | **26.9%** |
| GSM8K test (human-written) | 100 | **0 / 100** |

Those headline numbers are inflated by template-heavy categories: unanswerable was 80% (test) / 100% (hard), conflicting_sources 30% / 100%. Arithmetic, dates, composite, and ARC grids were at or near 0%. The seed often produces a fluent worked-solution *shape* with **wrong arithmetic, invented names, and leftover template phrasing** ("Trying every order of the 3 people…"). It learned the surface of the templates, not the procedure.

## How to load

`chat.py` and `seed_pool.py` are Seed v4 loaders. Use the v2 helper:

```bash
python v2/chat_v2.py organs/seeds/reason1-cael-undertrained.json
python v2/chat_v2.py organs/seeds/reason1-cael-undertrained.json --prompt "User: Janet has 12 apples and gives away 5. Then she buys 9 more. How many apples does she have now?\nAssistant:" --n 120
```

Or export your own checkpoint the same way:

```bash
python scripts/export_seed_v2.py path/to/best_ce.pt organs/seeds/myseed \
    --name myseed --tokenizer-dir tokenizers/seed-bpe8192-r1
```

Data prep, tokenizer training, mix file, and the exact-match eval live under [`recipes/reason1_cael/`](../../recipes/reason1_cael/).

## Next

A grown **25–50M** reasoning seed (still dynamically grown, still under the 2 GB weight cap), with exact-match tracked on human-written problems as the seed scales — not just held-out CE on templated CAEL.
