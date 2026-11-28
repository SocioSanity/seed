# Data sources and citations

Seed's code is Apache 2.0 (see [LICENSE](LICENSE)). The datasets listed here are third-party and keep their own licenses. The data files are not in this repo.

## CAEL Verified Reasoning

- **Name:** CAEL Verified Reasoning: computed-answer reasoning problems with a verifier
- **Author:** Vovala14 (CAEL)
- **Link:** https://huggingface.co/datasets/vynly/cael-verified-reasoning
- **License:** CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Commercial use is allowed with attribution.
- **What it is:** about 2 million templated reasoning problems (arithmetic, logic lineups, calendar puzzles, multi-step word problems, ARC-style grids). Every answer is computed by an exact solver, and each row has a worked solution.
- **How Seed uses it:** core training data (revision 1.1.1) for the **UNDERTRAINED / experimental** Seed v2.0 organ [`reason1-cael-undertrained`](organs/seeds/reason1-cael-undertrained.md). Rows are converted to question, worked solution, and final-answer text. Official `test` and `hard` splits were held out. No endorsement by the dataset author is implied.

Please cite it as:

```bibtex
@misc{cael_verified_reasoning_2026,
  title  = {CAEL Verified Reasoning: computed-answer reasoning problems with a verifier},
  author = {Vovala14 (CAEL)},
  year   = {2026},
  url    = {https://huggingface.co/datasets/vynly/cael-verified-reasoning}
}
```

## GSM8K

- **Name:** GSM8K (Grade School Math 8K)
- **Link:** https://huggingface.co/datasets/openai/gsm8k
- **License:** MIT
- **How Seed uses it:** a 4% share of the `reason1-cael-undertrained` mix (train split) and a held-out human-written exact-match eval (test split).

## SmolTalk

- **Name:** SmolTalk
- **Link:** https://huggingface.co/datasets/HuggingFaceTB/smoltalk
- **License:** Apache-2.0
- **How Seed uses it:** an 8% share of the `reason1-cael-undertrained` mix, as general conversational English so the seed is not only trained on templated word problems.

## FineWeb-Edu

- **Name:** FineWeb-Edu
- **Link:** https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- **License:** ODC-BY 1.0
- **How Seed uses it:** a 6% share of the `reason1-cael-undertrained` mix, as general educational English.
