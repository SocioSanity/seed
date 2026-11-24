# Data

This folder is optional. The shared tokenizer also lives at `../tokenizers/seed-bpe4096-v1/`.

`chat.py` and `seed_v4.py` will look for `data/tokenizer_4096.json` (a copy of the same tokenizer) if you put one here.

To build `train.bin` and `valid.bin` from TinyStories or your own files (those binaries are gitignored):

```bash
python prep_data.py --train-text path/to/TinyStoriesV2-GPT4-train.txt --valid-text path/to/TinyStoriesV2-GPT4-valid.txt --out data
python prep_data.py --data-dir path/to/your_texts --out data
```
