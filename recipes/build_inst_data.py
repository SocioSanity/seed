r"""seed_inst1 data (2026-11-24 CST): instruction following. smoltalk (local D:\datasets\smoltalk\smoltalk_train.txt, first ~250 MB) converted
from 'User:/Assistant:' to our <|user|>/<|assistant|> chat format; run4 instruct.txt is mixed in by the launcher (training region only).
smoltalk_valid.txt is never used for training (held-out bpb)."""
import re
src = r"D:\datasets\smoltalk\smoltalk_train.txt"; out = r"D:\seed_v4\seed_inst1\data\inst_smol.txt"; E = "<|endoftext|>"
raw = open(src, encoding="utf-8").read(270_000_000)
docs = re.split(r"\n\n(?=User: )", raw)   # conversations are separated by a blank line (458,341 docs per meta.json)
print("docs", len(docs))
n = 0
with open(out, "w", encoding="utf-8", newline="\n") as f:
    for d in docs[:-1]:
        parts = re.split(r"(?m)^(User|Assistant): ", d.strip())
        if len(parts) < 5 or parts[0].strip(): continue   # needs >= 1 user + 1 assistant turn
        s = ""; ok = True
        for role, txt in zip(parts[1::2], parts[2::2]):
            s += ("<|user|>" if role == "User" else "<|assistant|>") + txt.strip() + "<|end|>\n"
        if ok and "<|assistant|>" in s: f.write(s.rstrip("\n") + E); n += 1
print("written", n)
