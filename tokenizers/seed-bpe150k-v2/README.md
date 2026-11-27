# Seed v2.0 tokenizer (150,000 ByteLevel BPE)

This is the official Seed v2.0 tokenizer. **Pool 5 and Seed v4 still use** `seed-bpe4096-v1`. Use this tokenizer only with `v2/seed_v2.py`.

- Vocab: 150,000
- Algorithm: ByteLevel BPE
- `tokenizer.json` SHA-256: `fc67f00b300e872734502b96eeee00440be6266bfe83338a164acfb7ca847d73`
- Single-digit splitting (`12345` -> `1 2 3 4 5`)
- Intact 4-space and 8-space indent tokens
- 13 special tokens: `<pad>`, `<unk>`, `<bos>`, `<eos>`, `<|system|>`, `<|user|>`, `<|assistant|>`, `<|tool|>`, `<|tool_result|>`, `<|think|>`, `<|/think|>`, `<|code|>`, `<|end|>`

Files in this folder: `tokenizer.json`, `vocab.json`, `merges.txt`, `manifest.json`, `metadata.json`, `benchmark.json`, `corpus_breakdown.json`.

Do not train a new 150k BPE if you want to share or mix Seed v2.0 specialists. They must all use this exact `tokenizer.json`.
