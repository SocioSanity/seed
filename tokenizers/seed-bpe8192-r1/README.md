# seed-bpe8192-r1 (specialist tokenizer)

8192-token ByteLevel BPE used only by the **UNDERTRAINED / experimental** Seed v2.0 organ [`reason1-cael-undertrained`](../../organs/seeds/reason1-cael-undertrained.md).

This is **not** the official Seed v2.0 tokenizer (`seed-bpe150k-v2`, 150k) and **not** the Pool 5 tokenizer (`seed-bpe4096-v1`). It copies the v2.0 pre-tokenizer (digits are split individually; the same 13 special tokens) but uses a smaller vocab so the factorized embedding table fits a CPU training run (`8192 × 128` instead of `150000 × 128`). Seeds trained on this tokenizer cannot be dropped into a 150k v2 pool or into Pool 5.

- Vocab: 8,192
- Algorithm: ByteLevel BPE
- `tokenizer.json` SHA-256: `edba7163850c77d284ae40a7f0aeaab981cfef2b7c0d300e23e5718bd0da445a`
- Special tokens: `<pad>`, `<unk>`, `<bos>`, `<eos>`, `<|system|>`, `<|user|>`, `<|assistant|>`, `<|tool|>`, `<|tool_result|>`, `<|think|>`, `<|/think|>`, `<|code|>`, `<|end|>`

Trained on a 5% sample of formatted CAEL Verified Reasoning train rows, GSM8K train (duplicated once), and 40 MB of SmolTalk + FineWeb-Edu. Rebuild with `recipes/reason1_cael/build_tokenizer.py` if you have the local data files.
