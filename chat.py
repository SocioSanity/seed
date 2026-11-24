"""Chat with any Seed v4 checkpoint (latest.pt, a snapshot, or a tree part), optionally with extra parts.

  python chat.py runs/run1/latest.pt                      # EMA weights, GPU if available, interactive
  python chat.py runs/run1/latest.pt --prompt "Once upon a time" --n 120
  python chat.py parts/exports\\tree_....json --parts organ_a.json organ_b.json --device cpu
Commands in the REPL: /temp 0.7  /topk 40  /topp 0.9  /n 200  /quit
"""
import argparse, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
import json
from seed_v4 import load_model_any, install_parts_full, find_tokenizer, read_bundle, install_bundle, load_socket, ORGAN_FORMATS, DEFAULT_SOCKET_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="latest.pt / snapshot .pt / tree part .json")
    ap.add_argument("--parts", nargs="*", default=[], help="organ or cell part manifests (.json) to plug in")
    ap.add_argument("--no-ema", action="store_true", help="use raw weights instead of the EMA average")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--prompt", default=None, help="one-shot: print a continuation and exit")
    ap.add_argument("--n", type=int, default=150); ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40); ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--rep-pen", type=float, default=1.1); ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--socket-dir", default=DEFAULT_SOCKET_DIR)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = "cuda" if (a.device == "cuda" or (a.device == "auto" and torch.cuda.is_available())) else "cpu"
    if a.seed is not None:
        torch.manual_seed(a.seed)
    model, info = load_model_any(a.model, use_ema=not a.no_ema, device=dev)
    bundles = [p for p in a.parts if json.loads(Path(p).read_text(encoding="utf-8")).get("format") in ORGAN_FORMATS]
    legacy = [p for p in a.parts if p not in bundles]
    added = install_parts_full(model, legacy) if legacy else []
    for b in bundles:   # portable organ bundles (seed-socket-v1): aligned, installed at full strength
        man, t = read_bundle(b)
        _, _, calib = load_socket(a.socket_dir)
        oid, rep = install_bundle(model, man, t, calib, align="auto")
        for l in model.layers:
            l.cells[model.organ_ids.index(oid)].alpha.fill_(1.0)
        model.to(dev)
        added.append(f"organ {man.get('name')}@{man.get('version')} as {oid}")
    model.eval()
    tok = find_tokenizer(model, a.tokenizer)
    eos = tok.token_to_id("<eos>")
    print(f"Seed {model.shape_str()} {model.n_params()/1e6:.2f}M params | {info['kind']} step {info['step']} | "
          f"{'EMA' if info['ema'] else 'raw'} weights | {dev}" + (f" | parts: {', '.join(added)}" if added else ""), flush=True)
    st = dict(temp=a.temp, top_k=a.top_k, top_p=a.top_p, n=a.n)

    def gen(text):
        ids = tok.encode(text).ids or [eos]
        x = torch.tensor([ids], device=dev)
        t0 = time.time()
        y = model.generate(x, st["n"], temp=st["temp"], top_k=st["top_k"], top_p=st["top_p"], rep_pen=a.rep_pen, stop_id=eos)
        new = y[0, len(ids):].tolist()
        return tok.decode(new), len(new) / max(1e-6, time.time() - t0)

    if a.prompt is not None:
        out, tps = gen(a.prompt)
        print(a.prompt + out); print(f"[{tps:.0f} tok/s]")
        return
    print("type a prompt (empty line = new story start, /quit to leave)")
    while True:
        try:
            s = input("> ")
        except (EOFError, KeyboardInterrupt):
            break
        if s.startswith("/"):
            p = s[1:].split()
            if not p or p[0] in ("quit", "q", "exit"):
                break
            if len(p) == 2 and p[0] in ("temp", "topk", "topp", "n"):
                key = {"topk": "top_k", "topp": "top_p"}.get(p[0], p[0])
                st[key] = (int if key in ("top_k", "n") else float)(p[1]); print(st)
            continue
        out, tps = gen(s or "Once upon a time")
        print((s or "Once upon a time") + out); print(f"[{tps:.0f} tok/s]")


if __name__ == "__main__":
    main()
