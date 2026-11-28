#!/usr/bin/env python3
"""Build reasoning-1 text corpora from CAEL Verified Reasoning (CC BY 4.0) + GSM8K (MIT).

Dedupes CAEL on core question text, blocks leakage from official test/hard into train,
and length-filters documents above --max-toks (default 480).
"""
from __future__ import annotations
import argparse, collections, glob, json, os, sys
from pathlib import Path

os.environ.setdefault("RAYON_NUM_THREADS", "4")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from cael_fmt import cael_doc, core_key, prompt_of, gsm_doc, EOT

COLS = ["id", "category", "task", "difficulty", "messages", "answer", "solution"]


def rows(path):
    pf = pq.ParquetFile(path)
    for b in pf.iter_batches(batch_size=8192, columns=COLS):
        yield from b.to_pylist()


def ntoks(tok, docs):
    return [len(e.ids) for e in tok.encode_batch(docs, add_special_tokens=False)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cael-raw", default=r"D:\seed_v2\data\cael\raw")
    ap.add_argument("--out-text", default=r"D:\seed_v2\data\cael\text")
    ap.add_argument("--gsm8k-dir", default=r"D:\seed_v2\data\gsm8k")
    ap.add_argument("--tokenizer", default=str(HERE.parents[1] / "tokenizers" / "seed-bpe8192-r1" / "tokenizer.json"))
    ap.add_argument("--max-toks", type=int, default=480)
    a = ap.parse_args()
    raw, out = Path(a.cael_raw), Path(a.out_text)
    out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(a.tokenizer)
    stats = collections.defaultdict(lambda: collections.Counter())

    evalkeys = set()
    for name, fn in [("test", "test-00000-of-00001.parquet"), ("hard", "hard-00000-of-00001.parquet")]:
        rs = list(rows(raw / fn))
        docs = [cael_doc(r) for r in rs]
        nt = ntoks(tok, docs)
        outf = {}
        ev = open(out / f"eval_{name}.jsonl", "w", encoding="utf8")
        for r, d, n in zip(rs, docs, nt):
            evalkeys.add(core_key(r["messages"]))
            stats[name]["rows"] += 1
            stats[name][r["category"]] += 1
            ev.write(json.dumps(dict(id=r["id"], category=r["category"], task=r["task"],
                                     difficulty=r["difficulty"], prompt=prompt_of(r["messages"]),
                                     answer=r["answer"], toks=n), ensure_ascii=False) + "\n")
            if name == "test" and n <= a.max_toks:
                if r["category"] not in outf:
                    outf[r["category"]] = open(out / f"val_{r['category']}.txt", "w", encoding="utf8", newline="\n")
                outf[r["category"]].write(d + EOT)
                stats["test_kept"][r["category"]] += 1
        ev.close()
        for o in outf.values():
            o.close()
    print("eval keys", len(evalkeys), flush=True)

    seen, tr = set(), {}
    for f in sorted(glob.glob(str(raw / "train-*.parquet"))):
        rs = list(rows(f))
        docs = [cael_doc(r) for r in rs]
        nt = ntoks(tok, docs)
        for r, d, n in zip(rs, docs, nt):
            c = r["category"]
            stats["train_in"][c] += 1
            k = core_key(r["messages"])
            if k in evalkeys:
                stats["train_dropped_leak_vs_test_hard"][c] += 1
                continue
            if k in seen:
                stats["train_dropped_dup_core"][c] += 1
                continue
            seen.add(k)
            if n > a.max_toks:
                stats["train_dropped_too_long"][c] += 1
                continue
            if c not in tr:
                tr[c] = open(out / f"train_{c}.txt", "w", encoding="utf8", newline="\n")
            tr[c].write(d + EOT)
            stats["train_kept"][c] += 1
            stats["train_tokens"][c] += n + 1
        print(f, dict(stats["train_kept"]), flush=True)
    for o in tr.values():
        o.close()

    gdir = Path(a.gsm8k_dir)
    g = pq.read_table(gdir / "train-00000-of-00001.parquet").to_pylist()
    with open(gdir / "gsm8k_train.txt", "w", encoding="utf8", newline="\n") as o:
        for r in g:
            o.write(gsm_doc(r["question"], r["answer"]) + EOT)
    gt = pq.read_table(gdir / "test-00000-of-00001.parquet").to_pylist()
    with open(gdir / "gsm8k_test.txt", "w", encoding="utf8", newline="\n") as o, \
            open(gdir / "eval_gsm8k_test.jsonl", "w", encoding="utf8") as e:
        for r in gt:
            o.write(gsm_doc(r["question"], r["answer"]) + EOT)
            e.write(json.dumps(dict(prompt="User: " + r["question"].strip() + "\nAssistant:",
                                    answer=r["answer"].split("####")[1].strip().replace(",", "")),
                               ensure_ascii=False) + "\n")
    stats["gsm8k"].update(train=len(g), test=len(gt))
    json.dump({k: dict(v) for k, v in stats.items()}, open(out / "prep_stats.json", "w"), indent=1)
    print(json.dumps({k: dict(v) for k, v in stats.items()}, indent=1))


if __name__ == "__main__":
    main()
