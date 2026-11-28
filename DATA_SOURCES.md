# Data sources and citations

Seed's code is Apache 2.0 (see [LICENSE](LICENSE)). The datasets listed here are third-party and keep their own licenses. The data files are not in this repo.

## CAEL Verified Reasoning

- **Name:** CAEL Verified Reasoning: computed-answer reasoning problems with a verifier
- **Author:** Vovala14 (CAEL)
- **Link:** https://huggingface.co/datasets/vynly/cael-verified-reasoning
- **License:** CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Commercial use is allowed with attribution.
- **What it is:** about 2 million templated reasoning problems (arithmetic, logic lineups, calendar puzzles, multi-step word problems, ARC-style grids). Every answer is computed by an exact solver, and each row has a worked solution.
- **How Seed uses it:** training data for a reasoning-focused seed (CPU training run). The data is converted to question, worked solution, final answer text. No endorsement by the dataset author is implied.

Please cite it as:

```bibtex
@misc{cael_verified_reasoning_2026,
  title  = {CAEL Verified Reasoning: computed-answer reasoning problems with a verifier},
  author = {Vovala14 (CAEL)},
  year   = {2026},
  url    = {https://huggingface.co/datasets/vynly/cael-verified-reasoning}
}
```
