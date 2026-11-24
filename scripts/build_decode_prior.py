"""Brace-bug fix (decode side): the frozen seed-socket-v1 anchor was taken from a TinyStories-only run, so tokens that
never occur in TinyStories ('{', '}', '>', '_', most single bytes...) ended up with near-identical anchor rows (cos ~0.999).
With the tied, frozen readout a socket seed cannot tell them apart: it gives the right total probability to the group,
but splits it ~evenly inside the group, so '{' is sampled as e.g. 'Ö' or a control byte.
This builds a per-group log-prior from real token frequencies (tool/reasoning + TinyStories text): inside each collapsed
group, logit_i += log(|G| * share_i). The group's total mass is unchanged; inside it, tokens follow their real frequency."""
import sys, json, numpy as np, torch
from pathlib import Path
from safetensors.torch import load_file
from tokenizers import Tokenizer
TOK = r"D:\seed_v4\tokenizers\seed-bpe4096-v1\tokenizer.json"
ANC = r"D:\seed_organs\socket\seed-socket-v1\anchor_emb.safetensors"
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else r"D:\seed_v4\decode_prior\seed-bpe4096-v1_cluster_prior.pt")
SRCS = [r"D:\seed_v4\run4_data\%s.txt" % n for n in ("wiki_tool", "glaive", "instruct", "calc", "tools_misc", "gsm8k")] + \
       [r"C:\Users\mcgar\Desktop\Seed\data_custom\gpu_brutal_slices\TinyStoriesV2-GPT4-train_500MB.txt"]
t = Tokenizer.from_file(TOK); V = t.get_vocab_size()
E = load_file(ANC)["emb"].float(); En = E / E.norm(dim=1, keepdim=True)
S = En @ En.T; S.fill_diagonal_(0)
adj = (S > 0.95)
# connected components of the >0.95 graph (only tokens with at least one such neighbour)
comp = -np.ones(V, dtype=int); groups = []
for i in range(V):
    if comp[i] >= 0 or not adj[i].any():
        continue
    stack, g = [i], []
    comp[i] = len(groups)
    while stack:
        u = stack.pop(); g.append(u)
        for v in torch.nonzero(adj[u]).flatten().tolist():
            if comp[v] < 0:
                comp[v] = len(groups); stack.append(v)
    groups.append(sorted(g))
cnt = np.zeros(V, dtype=np.float64); big = np.zeros((V, V), dtype=np.float32)
for p in SRCS:
    with open(p, "rb") as f:
        size = Path(p).stat().st_size
        for k in range(8):   # 8 x 512 KB spread over each file
            f.seek(int(size * k / 8)); raw = f.read(512 * 1024).decode("utf-8", errors="ignore").replace("<|endoftext|>", "\n")
            for e in t.encode_batch(raw.split("\n\n")):
                ids = np.array(e.ids, dtype=np.int64); np.add.at(cnt, ids, 1)
                if len(ids) > 1: np.add.at(big, (ids[:-1], ids[1:]), 1)
bias = torch.zeros(V)
info = []
for g in groups:
    c = cnt[g] + 0.5; share = c / c.sum()
    b = np.log(len(g) * share); bias[g] = torch.tensor(b, dtype=torch.float32)
    top = sorted(zip(c, g), reverse=True)[:6]
    info.append(dict(size=len(g), top=[(t.id_to_token(i), int(cc)) for cc, i in top]))
OUT.parent.mkdir(parents=True, exist_ok=True)
man = json.loads(Path(r"D:\seed_v4\tokenizers\seed-bpe4096-v1\seed_tokenizer_manifest.json").read_text())
# bigram version: inside a group, share(i | previous token) = (c(prev,i) + 5*uni_i) / (c(prev,G) + 5)
G = sorted(set(i for g in groups for i in g)); gi = {i: k for k, i in enumerate(G)}
table = torch.zeros(V, len(G))
for g in groups:
    c = cnt[g] + 0.5; uni = c / c.sum(); cols = [gi[i] for i in g]
    bg = big[:, g].astype(np.float64)
    sh = (bg + 5 * uni[None, :]) / (bg.sum(1, keepdims=True) + 5)
    table[:, cols] = torch.tensor(np.log(len(g) * sh), dtype=torch.float32)
torch.save(dict(bias=bias, group_ids=torch.tensor(G), bigram=table.half(), tokenizer_sha256=man["sha256"], anchor_file_note="seed-socket-v1 anchor a8438384b2c1b973",
                groups=[g for g in groups], note=__doc__), OUT)
print(f"{len(groups)} collapsed groups, {sum(len(g) for g in groups)} tokens; saved {OUT}")
for d in info: print(d)
print("bias '{'", float(bias[t.token_to_id('{')]), "'}'", float(bias[t.token_to_id('}')]), "'Ö'", float(bias[t.encode('Ö').ids[0]]))


