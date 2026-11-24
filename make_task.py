"""Micro-task for the transfer test: 'Zorbish' translation (a made-up language). One or two English words
are translated word by word into invented words. A seed that never saw Zorbish scores ~0 on it."""
import json, random, sys
from pathlib import Path
EN = ["cat", "dog", "sun", "tree", "red", "big", "ball", "girl", "boy", "happy", "fish", "bird", "cake", "park", "blue", "box",
      "moon", "car", "hat", "milk", "frog", "duck", "star", "rain"]
ZB = ["zib", "mor", "tusk", "vel", "quon", "drap", "kelp", "wix", "obu", "snarf", "plim", "gex", "yod", "brum", "fleck", "zant",
      "hoop", "trell", "quib", "noz", "fip", "gral", "vosk", "lumb"]
D = dict(zip(EN, ZB))


def ex(r):
    ws = r.sample(EN, r.choice([1, 1, 2]))
    return {"prompt": "Zorbish: " + " ".join(ws) + " =>", "answer": " " + " ".join(D[w] for w in ws) + "."}


out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
r = random.Random(7)
with open(out / "zorbish_train.jsonl", "w") as f:
    for _ in range(20000):
        f.write(json.dumps(ex(r)) + "\n")
r = random.Random(12345)
with open(out / "zorbish_eval.jsonl", "w") as f:
    for _ in range(256):
        f.write(json.dumps(ex(r)) + "\n")
print("ok", out)
