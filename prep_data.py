"""Seed data prep: build token files (and optionally a tokenizer) from TinyStories or your own files.

Writes uint16 token files: data/train.bin, data/valid.bin, plus meta.json.

To grow a seed that can join the published Seed Pool, pass the official tokenizer
instead of training a new one:

    python prep_data.py --data-dir path/to/your_texts --out data \
        --tokenizer tokenizers/seed-bpe4096-v1/tokenizer.json

Training a fresh BPE (the default when --tokenizer is omitted) produces a seed
that is *not* compatible with seed-socket-v1 or the published organs.
"""
import os
os.environ.setdefault("RAYON_NUM_THREADS", "3")  # keep tokenizer threads low
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
import argparse, json, time
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL = ["<pad>", "<unk>", "<bos>", "<eos>"]


def read_docs(path, mb):
    with open(path, "rb") as f:
        raw = f.read(int(mb * 2**20))
    text = raw.decode("utf-8", errors="ignore")
    docs = [d.strip() for d in text.split("<|endoftext|>")]
    docs = [d for d in docs if len(d) > 20]
    return docs[:-1] if len(docs) > 1 else docs  # last one may be cut off


TEXT_EXT = {".txt", ".md", ".markdown", ".rst", ".text"}


def read_pdf(path):
    try:
        from pypdf import PdfReader
    except ImportError:
        print(f"  skip {path.name}: install pypdf for PDF support"); return ""
    try:
        return "\n\n".join((pg.extract_text() or "") for pg in PdfReader(str(path)).pages)
    except Exception as e:
        print(f"  skip {path.name}: {e}"); return ""


def folder_docs(folder, max_mb=500.0, chunk_chars=2000):
    """Every .txt/.md/.pdf under a folder -> list of ~2,000-character documents."""
    folder = Path(folder); docs, total, files = [], 0, 0
    for f in sorted(folder.rglob("*")):
        ext = f.suffix.lower()
        if not f.is_file() or (ext not in TEXT_EXT and ext != ".pdf"):
            continue
        text = read_pdf(f) if ext == ".pdf" else f.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            continue
        files += 1; total += len(text)
        buf = ""
        for line in text.replace("\r\n", "\n").split("\n"):  # PDFs rarely have blank lines
            line = line.strip()
            if not line:
                if len(buf) >= chunk_chars // 2:
                    docs.append(buf); buf = ""
                elif buf:
                    buf += "\n\n"
                continue
            buf = f"{buf}\n{line}" if buf and not buf.endswith("\n\n") else buf + line
            if len(buf) >= chunk_chars:
                docs.append(buf); buf = ""
        if buf.strip():
            docs.append(buf.strip())
        if total > max_mb * 2**20:
            print(f"  reached {max_mb} MB cap, stopping"); break
    print(f"folder {folder}: {files} files, {total/2**20:.1f} MB text, {len(docs):,} chunks")
    return docs


def load_or_train_tokenizer(train_docs, out, vocab, tokenizer_path=None):
    """Load an existing tokenizer (official seed-bpe4096-v1, etc.) or train a new BPE."""
    out = Path(out)
    if tokenizer_path:
        src = Path(tokenizer_path)
        if not src.is_file():
            raise SystemExit(f"tokenizer not found: {src}")
        tok = Tokenizer.from_file(str(src))
        tok_path = src
        print("using existing tokenizer", tok_path, "vocab", tok.get_vocab_size())
        return tok, tok_path
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(train_docs, trainers.BpeTrainer(vocab_size=vocab, special_tokens=SPECIAL, min_frequency=2,
                                                           initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok_path = out / f"tokenizer_{vocab}.json"
    tok.save(str(tok_path))
    print("tokenizer saved", tok_path)
    return tok, tok_path


def build_from_folder(folder, out, vocab=4096, valid_every=20, tokenizer_path=None):
    """Tokenizer + train/valid token files from your own files. Returns meta dict."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    docs = folder_docs(folder)
    if len(docs) < 2:
        raise SystemExit(f"Not enough text found in {folder} (need .txt, .md or .pdf files)")
    valid = docs[::valid_every] if len(docs) >= valid_every else docs[:1]
    vset = set(id(d) for d in valid)
    train = [d for d in docs if id(d) not in vset] or docs
    tok, tok_path = load_or_train_tokenizer(train, out, vocab, tokenizer_path)
    eos = tok.token_to_id("<eos>")
    tr, va = encode(tok, train, eos), encode(tok, valid, eos)
    if va.size < 2048:  # small library: hold out the last 5% of tokens instead
        allt = encode(tok, docs, eos); cut = max(300, int(allt.size * 0.05))
        tr, va = allt[:-cut], allt[-cut:]
    tr.tofile(out / "train.bin"); va.tofile(out / "valid.bin")
    meta = dict(vocab=tok.get_vocab_size(), tokenizer=str(tok_path), train_tokens=int(tr.size), valid_tokens=int(va.size),
                train_docs=len(train), valid_docs=len(valid), source_folder=str(Path(folder).resolve()),
                seconds=round(time.time() - t0, 1))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return meta


def encode(tok, docs, eos):
    out = []
    for i in range(0, len(docs), 2000):
        for enc in tok.encode_batch(docs[i:i + 2000]):
            out.extend(enc.ids)
            out.append(eos)
    vocab = tok.get_vocab_size()
    dtype = np.uint16 if vocab <= 65535 else np.uint32
    return np.asarray(out, dtype=dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-text", default=None, help="path to training text (e.g. TinyStoriesV2-GPT4-train.txt)")
    ap.add_argument("--valid-text", default=None, help="path to validation text")
    ap.add_argument("--train-mb", type=float, default=100.0)
    ap.add_argument("--tok-mb", type=float, default=25.0)
    ap.add_argument("--valid-mb", type=float, default=4.0)
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--out", default="data")
    ap.add_argument("--data-dir", default=None, help="folder of your own .txt/.md/.pdf files to train on instead of TinyStories")
    ap.add_argument("--tokenizer", default=None, help="use this existing tokenizer.json (e.g. tokenizers/seed-bpe4096-v1/tokenizer.json) instead of training a new BPE. Required for seeds that should join the published pool.")
    a = ap.parse_args()
    if a.data_dir:
        build_from_folder(a.data_dir, a.out, a.vocab, tokenizer_path=a.tokenizer); return
    if not a.train_text or not a.valid_text:
        raise SystemExit("pass --train-text and --valid-text, or --data-dir FOLDER")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    docs = read_docs(a.train_text, a.train_mb)
    print(f"train docs: {len(docs):,}")
    if a.tokenizer:
        tok, tok_path = load_or_train_tokenizer(docs, out, a.vocab, a.tokenizer)
    else:
        tok_path = out / f"tokenizer_{a.vocab}.json"
        if tok_path.exists():
            tok = Tokenizer.from_file(str(tok_path)); print("reusing", tok_path)
        else:
            tok, tok_path = load_or_train_tokenizer(docs[:int(len(docs) * min(1.0, a.tok_mb / a.train_mb))], out, a.vocab, None)
    eos = tok.token_to_id("<eos>")
    tr_ids = encode(tok, docs, eos); tr_ids.tofile(out / "train.bin")
    vdocs = read_docs(a.valid_text, a.valid_mb)
    va_ids = encode(tok, vdocs, eos); va_ids.tofile(out / "valid.bin")
    meta = dict(vocab=tok.get_vocab_size(), tokenizer=str(tok_path), train_tokens=int(tr_ids.size),
                valid_tokens=int(va_ids.size), train_docs=len(docs), valid_docs=len(vdocs),
                source_train=a.train_text, source_valid=a.valid_text, train_mb=a.train_mb,
                seconds=round(time.time() - t0, 1))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
