r"""seed_c1 data (2026-11-24 CST): C ONLY. Colby's C (colby_code train split: Joker/Joker15 bootloader-kernel sources, atlas-apollo, shattered-dao, msys home
projects) + public C from the local MSYS2 toolchain headers (mingw64 + ucrt64 include trees, deduped; mostly API declarations - lower quality, noted).
Colby's held-out 10% (colby_code\lang\c\heldout.txt) is the eval set and is never used here."""
import os, re, hashlib
E = "<|endoftext|>"; OUT = r"D:\seed_v4\seed_c1\data"; seen = set(); n = b = 0
with open(os.path.join(OUT, "public_c_headers.txt"), "w", encoding="utf-8", newline="\n") as f:
    for root in (r"C:\msys64\mingw64\include", r"C:\msys64\ucrt64\include"):
        for dp, ds, fs in os.walk(root):
            for fn in fs:
                if not fn.endswith((".h", ".c")): continue
                try: t = open(os.path.join(dp, fn), encoding="utf-8", errors="ignore").read()
                except Exception: continue
                ls = t.split("\n")
                if len(t) < 300 or len(t) > 300_000 or max(map(len, ls)) > 600: continue
                h = hashlib.sha1(re.sub(r"\s+", "", t).encode()).hexdigest()
                if h in seen: continue
                seen.add(h); rel = os.path.relpath(os.path.join(dp, fn), root).replace("\\", "/")
                f.write(f"/* file: {rel} */\n{t}{E}"); n += 1; b += len(t)
print("public headers", n, "MB", round(b / 2**20, 1))
c = open(r"D:\seed_v4\colby_code\lang\c\train.txt", encoding="utf-8").read() + open(r"D:\seed_v4\colby_code\lang\c_header\train.txt", encoding="utf-8").read()
open(os.path.join(OUT, "colby_c.txt"), "w", encoding="utf-8", newline="\n").write(c); print("colby C MB", round(len(c.encode()) / 2**20, 2))
