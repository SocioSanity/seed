# pool2 eval scripts

These are the exact scripts used for the v0.3.0 numbers, as they ran on the author's Windows PC. Paths are absolute (D:\...) and point at local data:
the run4 data files, the local Wikipedia lead index, the Ollama GGUF blob and a separate venv for the HF bpb run. Edit the paths to run them yourself.

- `test_pool2.py` — bpb sets, 20 facts, 15 held-out obscure facts, 10 math, stories, speed, router weights by region (greedy)
- `test_qwen3b_fair.py` — Qwen2.5-Coder-3B (qwen2.5-coder:3b Q4_K_M, llama.cpp, greedy): our format + 3 examples, native tool template, closed book
- `qwen_bpb_hf.py` — Qwen bits per byte from HF weights (fp32), same documents and bytes as the seeds
