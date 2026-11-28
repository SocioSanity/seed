#!/usr/bin/env python3
"""Train the 8192 ByteLevel BPE used by reason1-cael-undertrained.

Same pre-tokenizer (digits split) and specials as tokenizers/seed-bpe150k-v2,
smaller vocab for CPU training. Requires local CAEL parquet, GSM8K, and optional
English .txt files (see --help). The published tokenizer is already in the repo.
"""
from __future__ import annotations
import argparse, json, os, random, sys, glob
from pathlib import Path

os.environ.setdefault("RAYON_NUM_THREADS", "4")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from cael_fmt import cael_doc, gsm_doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE.parents[1] / "tokenizers" / "seed-bpe8192-r1"))
    ap.add_argument("--cael-raw", default=r"D:\seed_v2\data\cael\raw")
    ap.add_argument("--gsm8k-train", default=r"D:\seed_v2\data\gsm8k\train-00000-of-00001.parquet")
    ap.add_argument("--v2-tokenizer", default=str(HERE.parents[1] / "tokenizers" / "seed-bpe150k-v2" / "tokenizer.json"))
    ap.add_argument("--english", action="append", default=[], help="path,mb  e.g. D:\\datasets\\smoltalk\\smoltalk_train.txt,20")
    ap.add_argument("--cael-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    if not a.english:
        a.english = [r"D:\datasets\smoltalk\smoltalk_train.txt,20", r"D:\datasets\fineweb_edu\fineweb_edu_train.txt,20"]
    random.seed(a.seed)
    docs = []
    for f in sorted(glob.glob(str(Path(a.cael_raw) / "train-*.parquet"))):
        for r in pq.read_table(f, columns=["messages", "solution", "answer"]).to_pylist():
            if random.random() < a.cael_frac:
                docs.append(cael_doc(r))
    n_cael = len(docs)
    g = pq.read_table(a.gsm8k_train).to_pylist()
    docs += [gsm_doc(r["question"], r["answer"]) for r in g] * 2
    eng = 0
    for spec in a.english:
        p, mb = spec.split(",")
        p, mb = Path(p), int(mb)
        size = p.stat().st_size
        chunk = 2 * 2**20
        with open(p, "rb") as fh:
            for _ in range(max(1, mb // 2)):
                fh.seek(random.randrange(0, max(1, size - chunk)))
                docs.append(fh.read(chunk).decode("utf8", "ignore"))
                eng += chunk
    print("cael docs", n_cael, "gsm", len(g), "english MB", eng / 2**20, flush=True)
    v2 = Tokenizer.from_file(a.v2_tokenizer)
    specials = [t.content for t in v2.get_added_tokens_decoder().values()]
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = v2.pre_tokenizer
    tok.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=8192, special_tokens=specials,
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                             min_frequency=2, show_progress=False)
    tok.train_from_iterator(docs, tr, length=len(docs))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok.save(str(out / "tokenizer.json"))
    print("vocab", tok.get_vocab_size(), "eos", tok.token_to_id("<eos>"), "->", out / "tokenizer.json")


if __name__ == "__main__":
    main()
