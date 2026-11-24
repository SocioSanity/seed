"""Seed v3: seed_organs.py + hunger, roots/branches, organizers, white blood cells (immune),
growing vocabulary, protected parts and MSG-style ramped cells. Every feature is behind its own
flag (default off); with all flags off this behaves like seed_organs.py.

Seed: a CPU-only language model that starts as ONE cell and grows itself.

A "cell" is a tiny parallel transformer block (2-head attention + small MLP + a per-token
gate) that reads the residual stream and adds its output back. A layer holds 1..N cells
side by side. The model starts with 1 layer x 1 cell and, whenever validation loss
plateaus, it:
  1. prunes cells that measurably do nothing (weights archived to disk, never deleted),
  2. duplicates its most useful cell (copy + noise, both copies' outputs halved, so the
     network computes the same function right after the split; the copies then specialize),
  3. or adds a new layer on top (copy of the best cell with its outputs zeroed, so it
     starts as an exact no-op) when the top layer is wide enough or widening stopped helping.
Every event is logged to events.jsonl with the loss right before/after (proof of
function preservation). Built for small PCs: no GPU, ~3 threads, a few hundred MB of RAM.
"""
import os
import argparse, copy, csv, json, math, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- model

DEV = torch.device("cpu")  # GPU build: set from --device in main()


class Cell(nn.Module):
    def __init__(self, d, n_head=2, head_dim=32, mlp=256):
        super().__init__()
        self.n_head, self.head_dim = n_head, head_dim
        self.ln = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * n_head * head_dim, bias=False)
        self.proj = nn.Linear(n_head * head_dim, d, bias=False)
        self.fc = nn.Linear(d, mlp, bias=False)
        self.down = nn.Linear(mlp, d, bias=False)
        self.gate = nn.Linear(d, 1)
        for w in (self.qkv, self.fc):
            nn.init.normal_(w.weight, std=0.02)
        for w in (self.proj, self.down):
            nn.init.normal_(w.weight, std=0.01)
        nn.init.zeros_(self.gate.weight); nn.init.constant_(self.gate.bias, 2.0)
        self.cid, self.parent, self.born = "c0", None, 0
        self.protected = False
        self.ramp, self.ramp_born, self.ramp_steps = 1.0, None, 0   # MSG-style output mask 0->1

    def forward(self, x):
        B, T, _ = x.shape
        h = self.ln(x)
        q, k, v = self.qkv(h).view(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, -1)
        out = self.proj(a) + self.down(F.gelu(self.fc(h)))
        y = torch.sigmoid(self.gate(h)) * out
        return y if self.ramp == 1.0 else y * self.ramp


class Router(nn.Module):
    """Organ router: one logit per token. Organ output is scaled by 0.5+sigmoid(logit); zero-init => scale 1.0 (no-op at birth)."""
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, 1)
        nn.init.zeros_(self.lin.weight); nn.init.zeros_(self.lin.bias)

    def forward(self, x):
        return self.lin(F.layer_norm(x, x.shape[-1:])).squeeze(-1)


class SeedLM(nn.Module):
    def __init__(self, vocab, d=128, ctx=128, cell_kw=None):
        super().__init__()
        self.vocab, self.d, self.ctx, self.cell_kw = vocab, d, ctx, dict(cell_kw or {})
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        nn.init.normal_(self.emb.weight, std=0.02); nn.init.normal_(self.pos.weight, std=0.01)
        self.layers = nn.ModuleList([nn.ModuleList([Cell(d, **self.cell_kw)])])
        self.ln_f = nn.LayerNorm(d)
        self.next_id = 1
        self.routers = nn.ModuleDict()
        self.organs = []          # list of dict(oid, layer, members=[cid], band, born, lift0, ...)
        self.organ_aux_w, self.organ_band = 0.0, 0.3
        self.base_vocab, self.n_new, self.nbase = vocab, 0, None   # grown vocab: ids >= base_vocab are merged tokens

    def alloc_new_vocab(self, n_new):
        """Allocate (uninitialised) room for n_new merged tokens: bigger input embedding, untied output rows
        for the new tokens, and an output bias over the whole vocab. Used on resume and by expand_vocab."""
        d = self.d
        old = self.emb.weight.data
        emb = nn.Embedding(self.base_vocab + n_new, d)
        emb.weight.data[:old.shape[0]] = old[:min(old.shape[0], self.base_vocab + n_new)]
        self.emb = emb
        self.head_new = nn.Parameter(torch.zeros(n_new, d))
        self.head_bias = nn.Parameter(torch.zeros(self.base_vocab + n_new))
        self.n_new = n_new
        self.vocab = self.base_vocab + n_new

    def set_organs(self, organs):
        self.organs = [dict(o) for o in organs]
        self.routers = nn.ModuleDict({o["oid"]: Router(self.d) for o in self.organs})

    def new_id(self):
        i = self.next_id; self.next_id += 1; return f"c{i}"

    def forward(self, idx, targets=None, skip=None, per_token=False, route=None, trace=None):
        B, T = idx.shape
        x = self.emb(idx) + self.pos(torch.arange(T, device=idx.device))
        skipset = set() if skip is None else ({skip} if isinstance(skip, tuple) else set(skip))
        aux_items = []
        for li, layer in enumerate(self.layers):
            outs = [None if (li, ci) in skipset else cell(x) for ci, cell in enumerate(layer)]
            if trace is not None:
                trace.append(torch.stack([(o.norm(dim=-1) if o is not None else torch.zeros(B, T)).flatten() for o in outs]))
            orgs = [o for o in self.organs if o["layer"] == li]
            if orgs:
                pos = {c.cid: ci for ci, c in enumerate(layer)}
                member = set()
                dx = 0
                for o in orgs:
                    r = self.routers[o["oid"]](x)
                    if route is not None:
                        route[o["oid"]] = torch.sigmoid(r.detach())
                    if targets is not None:
                        aux_items.append((o, r))
                    m = (0.5 + torch.sigmoid(r)).unsqueeze(-1)
                    tot = 0
                    for cid in o["members"]:
                        ci = pos.get(cid)
                        if ci is None:
                            continue
                        member.add(ci)
                        if outs[ci] is not None:
                            tot = tot + outs[ci]
                    if not isinstance(tot, int):
                        dx = dx + m * tot
                for ci, ot in enumerate(outs):
                    if ci not in member and ot is not None:
                        dx = dx + ot
            else:
                dx = 0
                for ot in outs:
                    if ot is not None:
                        dx = dx + ot
            x = x + dx
        hf = self.ln_f(x)
        if self.n_new:  # base tokens: tied head; merged tokens: own output rows; plus a vocab bias
            logits = torch.cat([hf @ self.emb.weight[:self.base_vocab].t(), hf @ self.head_new.t()], -1) + self.head_bias
        else:
            logits = hf @ self.emb.weight.t()  # tied head
        if targets is None:
            return logits
        if not per_token and not (aux_items and self.organ_aux_w > 0 and self.training):
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        pt = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none").view(B, T)
        if per_token:
            return pt
        loss = pt.mean()
        flat = pt.detach().flatten()
        aux = 0
        for o, r in aux_items:  # organs get first pick of the hardest tokens: band 0 = hardest 30%, band 1 = next 30%...
            tgt = band_mask(flat, o["band"], self.organ_band).view(B, T).float()
            aux = aux + F.binary_cross_entropy_with_logits(r, tgt)
        return loss + self.organ_aux_w * aux / len(aux_items)

    def structure(self):
        return [[dict(cid=c.cid, parent=c.parent, born=c.born, protected=bool(getattr(c, "protected", False)),
                      ramp_born=getattr(c, "ramp_born", None), ramp_steps=getattr(c, "ramp_steps", 0),
                      ramp=float(getattr(c, "ramp", 1.0))) for c in layer] for layer in self.layers]

    def shape_str(self):
        base = " | ".join(f"L{li}:[" + ",".join(c.cid + ("*" if getattr(c, "protected", False) else "") for c in layer) + "]" for li, layer in enumerate(self.layers))
        if self.organs:
            base += " ; organs: " + ", ".join(f"{o['oid']}@L{o['layer']}[" + "+".join(o["members"]) + f"]b{o['band']}" for o in self.organs)
        return base

    def n_cells(self):
        return sum(len(l) for l in self.layers)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_structure(cls, vocab, d, ctx, cell_kw, structure, next_id):
        m = cls(vocab, d, ctx, cell_kw)
        m.layers = nn.ModuleList()
        for layer in structure:
            ml = nn.ModuleList()
            for meta in layer:
                c = Cell(d, **m.cell_kw); c.cid, c.parent, c.born = meta["cid"], meta["parent"], meta["born"]
                c.protected = bool(meta.get("protected", False))
                c.ramp_born, c.ramp_steps, c.ramp = meta.get("ramp_born"), meta.get("ramp_steps", 0), float(meta.get("ramp", 1.0))
                ml.append(c)
            m.layers.append(ml)
        m.next_id = next_id
        return m

    @torch.no_grad()
    def generate(self, idx, n, temp=0.8, top_k=40):
        for _ in range(n):
            logits = self(idx[:, -self.ctx:])[:, -1, :] / temp
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(F.softmax(logits, -1), 1)
            idx = torch.cat([idx, nxt], 1)
        return idx

# --------------------------------------------------------------------------- growth ops

def rebuild_optimizer(model, old_opt, lr, wd, state_map=None):
    """New AdamW over current params; carry over moments for surviving params and
    clone the parent's moments into duplicated params (state_map: new_param -> old_param)."""
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and "pos" not in n]
    no_decay = [p for n, p in model.named_parameters() if not (p.dim() >= 2 and "pos" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": wd},
                             {"params": no_decay, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.98))
    if old_opt is not None:
        state_map = state_map or {}
        for p in model.parameters():
            src = state_map.get(p, p)
            st = old_opt.state.get(src)
            if st:
                opt.state[p] = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in st.items()}
    return opt


@torch.no_grad()
def duplicate_cell(model, li, ci, step, noise=0.02, exact=False):
    parent = model.layers[li][ci]
    child = copy.deepcopy(parent)
    child.cid, child.parent, child.born = model.new_id(), parent.cid, step
    child.protected = False  # a protected parent's twin is a normal cell
    for c in (parent, child):  # split outputs: parent/2 + child/2 == old parent
        c.proj.weight.mul_(0.5); c.down.weight.mul_(0.5)
    if exact:  # v3 --exact-split: antisymmetric output noise (+E / -E) cancels exactly, still breaks symmetry
        for wp, wc in ((parent.proj.weight, child.proj.weight), (parent.down.weight, child.down.weight)):
            e = torch.randn_like(wp) * noise * wp.std()
            wp.add_(e); wc.sub_(e)
    else:
        for w in (child.qkv.weight, child.fc.weight):  # break symmetry on the input side only (approximate)
            w.add_(torch.randn_like(w) * noise * w.std())
        child.gate.weight.add_(torch.randn_like(child.gate.weight) * 0.02)
    for o in model.organs:  # a twin inherits the parent's organ membership
        if o["layer"] == li and parent.cid in o["members"]:
            o["members"].append(child.cid)
    model.layers[li].append(child)
    pmap = dict(zip(child.parameters(), parent.parameters()))
    return child, pmap


@torch.no_grad()
def add_layer(model, step, template, ramp_steps=0):
    cell = copy.deepcopy(template)
    cell.cid, cell.parent, cell.born = model.new_id(), template.cid, step
    cell.protected = False
    if ramp_steps > 0:  # MSG-style: keep the template's weights, fade the output in 0 -> 1 (exact no-op at birth)
        cell.ramp, cell.ramp_born, cell.ramp_steps = 0.0, step, ramp_steps
    else:
        cell.proj.weight.zero_(); cell.down.weight.zero_()  # exact no-op at birth
        cell.ramp, cell.ramp_born, cell.ramp_steps = 1.0, None, 0
    model.layers.append(nn.ModuleList([cell]))
    return cell


def prune_cell(model, li, ci, archive_dir, step, reason):
    cell = model.layers[li][ci]
    archive_dir.mkdir(parents=True, exist_ok=True)
    path = archive_dir / f"pruned_{cell.cid}_L{li}_step{step}.pt"
    torch.save(dict(state=cell.state_dict(), cid=cell.cid, parent=cell.parent, born=cell.born,
                    layer=li, step=step, reason=reason), path)
    del model.layers[li][ci]
    return path

# --------------------------------------------------------------------------- data / eval

class Tokens:
    def __init__(self, path):
        self.a = np.memmap(path, dtype=np.uint16, mode="r")

    def batch(self, B, T, rng):
        if len(self.a) < T + 2:
            raise SystemExit(f"Token file too small ({len(self.a)} tokens); add more text.")
        s = rng.integers(0, len(self.a) - T - 1, size=B)
        x = np.stack([self.a[i:i + T] for i in s]).astype(np.int64)
        y = np.stack([self.a[i + 1:i + T + 1] for i in s]).astype(np.int64)
        return torch.from_numpy(x).to(DEV), torch.from_numpy(y).to(DEV)


class TextMix:
    """Weighted mix of raw UTF-8 text files, streamed (seek+read of a few KB per sample, never
    loaded or memory-mapped, so RAM stays flat). The last --mix-holdout of every file is never
    trained on and is used for the val_mix set. '<|endoftext|>' becomes the <eos> token, which is
    how the prepared TinyStories train.bin was built. The byte-level BPE can encode any text."""
    SEP = "<|endoftext|>"

    def __init__(self, specs, tok, eos, holdout=0.01, read_bytes=768, train_only=(), rotate_dir=None, rotate_share=0.0, rotate_every=200):
        self.tok, self.eos, self.rb = tok, eos, read_bytes
        self.rot_dir, self.rot_share, self.rot_every, self._rot_calls, self.rot_names = rotate_dir, float(rotate_share), int(rotate_every), 0, ()
        self.src = []
        for sp, val in [(x, True) for x in specs] + [(x, False) for x in (train_only or [])]:
            w, p = sp.split(":", 1)
            size = os.path.getsize(p)
            cut = int(size * (1 - holdout))
            self.src.append(dict(path=p, w=float(w), size=size, cut=cut, f=open(p, "rb"), name=Path(p).stem, val=val))
        tot = sum(s["w"] for s in self.src)
        self.p = np.array([s["w"] / tot for s in self.src])
        # val_mix uses only the --mix sources (train-only sources are appended after them), with the
        # shares renormalised among them, so val_mix stays identical when train-only data is added.
        self.val_idx = np.array([i for i, s in enumerate(self.src) if s["val"]])
        vw = np.array([self.src[i]["w"] for i in self.val_idx])
        self.p_val = vw / vw.sum()
        self.n_static = len(self.src)
        if self.rot_dir:
            self.refresh_rotation(first=True)

    def refresh_rotation(self, first=False):
        """Data rotation: the --rotate-dir folder holds staged chunks (*.txt, trained on only, never in val_mix).
        A background stager (rotation_stager.py) adds new chunks and retires old ones (marks them with a
        '<chunk>.retire' file). This re-scans the folder, closes retired/removed chunks (so the stager can delete
        them) and opens new ones, splitting --rotate-share evenly across the chunks present. Static sources and
        val_mix are untouched, so scores stay comparable. If the folder is empty its share is simply skipped."""
        try:
            names = sorted(f for f in os.listdir(self.rot_dir)
                           if f.endswith(".txt") and not os.path.exists(os.path.join(self.rot_dir, f + ".retire"))
                           and os.path.getsize(os.path.join(self.rot_dir, f)) > 2**20)
        except OSError:
            return
        if tuple(names) == tuple(self.rot_names) and not first:
            return
        old = {s["name"]: s for s in self.src[self.n_static:]}
        new = []
        for n in names:
            if n in old:
                new.append(old.pop(n)); continue
            pth = os.path.join(self.rot_dir, n); size = os.path.getsize(pth)
            try:
                new.append(dict(path=pth, w=0.0, size=size, cut=size, f=open(pth, "rb"), name=n, val=False))
            except OSError:
                pass
        for s_ in old.values():
            try: s_["f"].close()
            except Exception: pass
        for s_ in new: s_["w"] = self.rot_share / max(1, len(new))
        self.src = self.src[:self.n_static] + new
        self.rot_names = tuple(s_["name"] for s_ in new)
        tot = sum(s_["w"] for s_ in self.src)
        self.p = np.array([s_["w"] / tot for s_ in self.src])
        print(f"[rotation] {len(new)} chunks in {self.rot_dir} ({sum(s_['size'] for s_ in new)/2**30:.2f} GB), share {self.rot_share/tot*100:.1f}%", flush=True)

    def _text(self, s, rng, held):
        lo, hi = (s["cut"], s["size"]) if held else (0, s["cut"])
        off = int(rng.integers(lo, max(lo + 1, hi - self.rb)))
        s["f"].seek(off); raw = s["f"].read(self.rb)
        sp = raw.find(b" ")  # start on a word boundary (also skips a split UTF-8 char)
        raw = raw[sp + 1:] if 0 <= sp < 64 else raw
        return raw.decode("utf-8", errors="ignore").replace("\r\n", "\n").replace("\r", "\n")

    def _ids(self, text):
        out = []
        for i, piece in enumerate(text.split(self.SEP)):
            if i: out.append(self.eos)
            if piece: out.extend(self.tok.encode(piece).ids)
        return out

    def batch(self, B, T, rng, held=False, which=None):
        if self.rot_dir and not held and which is None:
            self._rot_calls += 1
            if self._rot_calls % self.rot_every == 0:
                self.refresh_rotation()
        which = rng.choice(len(self.src), size=B, p=self.p) if which is None else which
        xs = []
        for k in which:
            s = self.src[k]; ids = []
            while len(ids) < T + 1:
                ids += self._ids(self._text(s, rng, held))
            xs.append(ids[:T + 1])
        a = torch.tensor(xs, dtype=torch.long, device=DEV)
        return a[:, :-1].contiguous(), a[:, 1:].contiguous()

    def val_batches(self, n, B, T, seed=777):
        """Fixed held-out batches; rows are split across sources in proportion to the shares."""
        r = np.random.default_rng(seed)
        cnt = np.floor(self.p_val * n * B).astype(int)
        cnt[np.argmax(self.p_val)] += n * B - cnt.sum()
        rows = np.repeat(self.val_idx, cnt); r.shuffle(rows)
        return [self.batch(B, T, r, held=True, which=rows[i * B:(i + 1) * B]) for i in range(n)]


@torch.no_grad()
def evaluate(model, batches, skip=None):
    """Mean loss. Once the vocab has grown, this is nats per BASE token (sum of losses / number of base-4096
    tokens the targets stand for), so numbers stay comparable with a model that never grew its vocab."""
    model.eval()
    if getattr(model, "nbase", None) is None:
        l = sum(model(x, y, skip=skip).item() for x, y in batches) / len(batches)
    else:
        tot, cnt = 0.0, 0
        for x, y in batches:
            tot += model(x, y, skip=skip, per_token=True).sum().item(); cnt += model.nbase[y].sum().item()
        l = tot / cnt
    model.train()
    return l


@torch.no_grad()
def usefulness(model, batches):
    """Loss increase when each cell is switched off (bigger = more useful)."""
    base = evaluate(model, batches)
    u = {}
    for li, layer in enumerate(model.layers):
        for ci in range(len(layer)):
            u[(li, ci)] = evaluate(model, batches, skip=(li, ci)) - base
    return base, u


@torch.no_grad()
def sibling_similarity(model, x):
    """Mean cosine similarity between cells in the same layer (1.0 = identical, lower = specialized)."""
    model.eval()
    h = model.emb(x) + model.pos(torch.arange(x.shape[1]))
    sims = []
    for layer in model.layers:
        outs = [c(h).flatten() for c in layer]
        for i in range(len(outs)):
            for j in range(i + 1, len(outs)):
                sims.append(F.cosine_similarity(outs[i], outs[j], dim=0).item())
        h = h + sum(outs_i.view_as(h) for outs_i in outs)
    model.train()
    return float(np.mean(sims)) if sims else None


def rss_mb():
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return mi.rss / 2**20, getattr(mi, "peak_wset", mi.rss) / 2**20
    except Exception:
        return float("nan"), float("nan")


def lower_priority(level="idle"):
    try:
        import psutil
        p = psutil.Process()
        if os.name == "nt":
            p.nice(psutil.IDLE_PRIORITY_CLASS if level == "idle" else psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            p.nice(15 if level == "idle" else 10)
        return True
    except Exception:
        return False

# --------------------------------------------------------------------------- organs

def band_mask(pt, band, w):
    """Tokens whose loss rank falls in difficulty band `band` (0 = hardest fraction w, 1 = next w, ...)."""
    hi = 1.0 - w * band
    lo = max(0.0, hi - w)
    m = pt >= torch.quantile(pt, lo)
    if band > 0:
        m = m & (pt < torch.quantile(pt, min(hi, 1.0)))
    return m


@torch.no_grad()
def ptl(model, batches, skip=None, route=False):
    """Flat per-token losses over batches (optionally with cells in `skip` switched off, and organ router outputs)."""
    model.eval()
    pts, rt = [], {}
    for x, y in batches:
        rd = {} if route else None
        pts.append(model(x, y, skip=skip, per_token=True, route=rd).flatten())
        if route:
            for k, v in rd.items():
                rt.setdefault(k, []).append(v.flatten())
    model.train()
    pt = torch.cat(pts)
    return (pt, {k: torch.cat(v) for k, v in rt.items()}) if route else pt


def niche_tokens(y, mask, tok, vocab, k=8):
    ca = torch.bincount(y, minlength=vocab).float(); cb = torch.bincount(y[mask], minlength=vocab).float()
    score = cb * (cb / ca.clamp(min=1)); score[ca < 10] = 0
    out = []
    for i in score.topk(k).indices.tolist():
        if score[i] > 0:
            out.append(tok.decode([i]).strip() or repr(tok.decode([i])))
    return out


def dissolve_organ(model, o, archive_dir, step, why):
    archive_dir.mkdir(parents=True, exist_ok=True)
    path = archive_dir / f"organ_{o['oid']}_dissolved_step{step}.pt"
    torch.save(dict(router=model.routers[o["oid"]].state_dict(), organ=o, step=step, reason=why), path)
    del model.routers[o["oid"]]
    model.organs = [x for x in model.organs if x["oid"] != o["oid"]]
    return path


def organ_protected(model, o):
    if o.get("protected"):
        return True
    return any(getattr(c, "protected", False) for c in model.layers[o["layer"]] if c.cid in o["members"]) if o["layer"] < len(model.layers) else False


def cell_protected(model, li, cell):
    if getattr(cell, "protected", False):
        return True
    return any(o.get("protected") and o["layer"] == li and cell.cid in o["members"] for o in model.organs)


def organ_on_prune(model, cid, li, archive_dir, step, event):
    """A pruned cell leaves its organ; an organ left with <2 cells dissolves (router archived)."""
    for o in list(model.organs):
        if o["layer"] == li and cid in o["members"]:
            o["members"] = [m for m in o["members"] if m != cid]
            if len(o["members"]) < 2 and not o.get("protected"):
                path = dissolve_organ(model, o, archive_dir, step, "member pruned")
                event("organ_dissolve", organ=o["oid"], layer=li, reason="member pruned", archived=str(path))


def organ_check(model, val_batches, tok, a, step, state, out, event, log):
    """Form / dissolve organs. Returns True if the parameter set changed (caller rebuilds the optimizer)."""
    changed = False
    y = torch.cat([yy.flatten() for _, yy in val_batches])
    pt, route = ptl(model, val_batches, route=True)
    base = pt.mean().item()
    # ---- existing organs: measure lift on their niche, dissolve if it faded
    scan = []
    for o in list(model.organs):
        li = o["layer"]
        idxs = {(li, ci) for ci, c in enumerate(model.layers[li]) if c.cid in o["members"]}
        mk = band_mask(pt, o["band"], a.organ_band)
        pts = ptl(model, val_batches, skip=idxs)
        lift = (pts[mk] - pt[mk]).mean().item(); rest = (pts[~mk] - pt[~mk]).mean().item()
        r = route[o["oid"]]; sel = (r[mk].mean() - r[~mk].mean()).item()
        o.update(lift=round(lift, 4), lift_rest=round(rest, 4), selectivity=round(sel, 4), size=len(idxs))
        scan.append(f"{o['oid']}(L{li} n={len(idxs)} lift {lift:.3f}/{rest:.3f} sel {sel:+.3f})")
        if not organ_protected(model, o) and step >= getattr(a, "organ_grace_until", 0) and step - o["born"] >= a.organ_min_age and (lift < a.organ_dissolve_frac * o["lift0"] or lift < 0.5 * a.organ_lift_min):
            before = base
            path = dissolve_organ(model, o, out / "archive", step, f"lift {lift:.4f} < {a.organ_dissolve_frac}*{o['lift0']}")
            event("organ_dissolve", organ=o["oid"], layer=li, members=o["members"], lift=round(lift, 4), lift0=o["lift0"],
                  reason="lift fell", archived=str(path), val_before=round(before, 4), val_after=round(evaluate(model, val_batches), 4))
            changed = True
    # ---- formation: co-activating groups of cells in one layer with measurable lift on a hard-token niche
    tr_all = []
    model.eval()
    with torch.no_grad():
        for xb, _ in val_batches[:4]:
            tr = []; model(xb, trace=tr); tr_all.append(tr)
    model.train()
    best = None
    for li, layer in enumerate(model.layers):
        orgs = [o for o in model.organs if o["layer"] == li]
        used = {c for o in orgs for c in o["members"]}
        free = [ci for ci, c in enumerate(layer) if c.cid not in used]
        used_bands = {o["band"] for o in orgs}
        if len(orgs) >= a.organ_max_per_layer or len(free) < 2 or len(layer) < 3:
            continue
        band = next(b for b in range(10) if b not in used_bands)
        acts = torch.cat([t[li] for t in tr_all], 1)[free]
        C = torch.corrcoef(acts)
        n = len(free)
        iu = [(C[i, j].item(), i, j) for i in range(n) for j in range(i + 1, n)]
        c0, i0, j0 = max(iu)
        scan.append(f"L{li} best coact {c0:.2f}")
        if c0 < a.organ_tau:
            continue
        grp = [i0, j0]
        for k in range(n):
            if k not in grp and np.mean([C[k, g].item() for g in grp]) > a.organ_tau and len(grp) + 1 < len(layer):
                grp.append(k)
        idxs = {(li, free[g]) for g in grp}
        mk = band_mask(pt, band, a.organ_band)
        pts = ptl(model, val_batches, skip=idxs)
        lift = (pts[mk] - pt[mk]).mean().item(); rest = (pts[~mk] - pt[~mk]).mean().item()
        ratio = lift / max(rest, 1e-6)
        scan.append(f"L{li} cand n={len(grp)} lift {lift:.3f} rest {rest:.3f} ratio {ratio:.2f}")
        if lift >= a.organ_lift_min and ratio >= a.organ_ratio and (best is None or lift > best[0]):
            best = (lift, rest, ratio, li, band, [layer[free[g]].cid for g in grp], c0, mk)
    if best:
        lift, rest, ratio, li, band, members, c0, mk = best
        oid = f"o{state.get('next_oid', 1)}"; state["next_oid"] = state.get("next_oid", 1) + 1
        o = dict(oid=oid, layer=li, members=members, band=band, born=step, lift0=round(lift, 4), lift=round(lift, 4),
                 lift_rest=round(rest, 4), ratio=round(ratio, 2), coact=round(c0, 3), size=len(members),
                 niche=niche_tokens(y, mk, tok, model.emb.num_embeddings))
        before = evaluate(model, val_batches)
        model.organs.append(o); model.routers[oid] = Router(model.d)
        event("organ_form", organ=oid, layer=li, members=members, band=band, niche_hard_tokens=o["niche"], coact=o["coact"],
              lift_on_niche=o["lift0"], lift_elsewhere=o["lift_rest"], niche_ratio=o["ratio"],
              val_before=round(before, 4), val_after=round(evaluate(model, val_batches), 4))
        changed = True
    log("organ scan: " + "; ".join(scan) if scan else "organ scan: nothing to scan")
    return changed

# --------------------------------------------------------------------------- v3: growing vocabulary

class _Enc:
    def __init__(self, ids):
        self.ids = ids


class GrowTok:
    """The base byte-level BPE (ids 0..base-1 never change) plus an ordered list of extra merges learned
    by Seed. Encoding = base encoding, then the extra merges applied in rank order (standard BPE), so the
    result is deterministic: the same merge list always gives the same ids on any machine."""

    def __init__(self, tok, merges=None):
        self.tok, self.base = tok, tok.get_vocab_size()
        self.merges, self.rank, self.nb = [], {}, [1] * self.base
        for m in merges or []:
            self.add(int(m[0]), int(m[1]))

    def add(self, a, b):
        new = self.base + len(self.merges)
        self.rank[(a, b)] = (len(self.merges), new)
        self.merges.append((a, b, new))
        self.nb.append(self.nb[a] + self.nb[b])
        return new

    def vocab_size(self):
        return self.base + len(self.merges)

    def apply(self, ids):
        if not self.rank or len(ids) < 2:
            return list(ids)
        ids = list(ids)
        rank = self.rank
        while len(ids) > 1:
            best = None
            for i in range(len(ids) - 1):
                r = rank.get((ids[i], ids[i + 1]))
                if r is not None and (best is None or r[0] < best[0]):
                    best = r
            if best is None:
                break
            pa, pb, new = self.merges[best[0]]
            out, i, n = [], 0, len(ids)
            while i < n:
                if i < n - 1 and ids[i] == pa and ids[i + 1] == pb:
                    out.append(new); i += 2
                else:
                    out.append(ids[i]); i += 1
            ids = out
        return ids

    def encode(self, text):
        return _Enc(self.apply(self.tok.encode(text).ids))

    def expand(self, ids):
        out = []
        for i in ids:
            i = int(i)
            if i < self.base:
                out.append(i)
            else:
                a, b, _ = self.merges[i - self.base]
                out.extend(self.expand([a, b]))
        return out

    def decode(self, ids):
        return self.tok.decode(self.expand(ids))

    def token_to_id(self, s):
        return self.tok.token_to_id(s)

    def nbase_tensor(self):
        return torch.tensor(self.nb, dtype=torch.long)

    def sha(self):
        import hashlib
        return hashlib.sha256(json.dumps([[a, b] for a, b, _ in self.merges]).encode()).hexdigest()[:16]


def base_tok_sha(tok):
    import hashlib
    return hashlib.sha256(tok.to_str().encode("utf-8")).hexdigest()[:16]


def count_pairs(rows, eos, vocab):
    """Adjacent-pair counts over recent training rows (int64 arrays). Pairs touching <eos> or special ids are skipped."""
    tot = {}
    left = np.zeros(vocab, dtype=np.int64)
    for r in rows:
        a, b = r[:, :-1].ravel(), r[:, 1:].ravel()
        ok = (a > eos) & (b > eos)
        a, b = a[ok], b[ok]
        np.add.at(left, a, 1)
        key = a.astype(np.int64) * vocab + b
        u, c = np.unique(key, return_counts=True)
        for k, n in zip(u.tolist(), c.tolist()):
            tot[k] = tot.get(k, 0) + n
    return tot, left


def pick_merges(tot, left, vocab, k, min_count):
    """Most frequent pairs first; ties broken by (a, b) ids so the order is deterministic. Pairs that share a
    token with an already-picked pair this round are skipped (keeps each split exact and the order standard)."""
    items = sorted(((-n, kk // vocab, kk % vocab) for kk, n in tot.items() if n >= min_count))
    used, out = set(), []
    for negn, a, b in items:
        if len(out) >= k:
            break
        if a in used or b in used or a == b:
            continue
        q = (-negn) / max(1, left[a])
        out.append((a, b, -negn, min(max(q, 1e-4), 0.999)))
        used.update((a, b))
    return out


@torch.no_grad()
def expand_vocab(model, gt, picks):
    """Add merged tokens. New token t=(a,b): input embedding = mean of its pieces; output row = a's output row;
    probability mass is SPLIT, not created: bias[t] = bias[a] + log q, bias[a] += log(1-q), where q = how often
    a is followed by b in recent text. So P(a) before == P(a) + P(ab) after, and every other token's
    probability is unchanged (exact up to float error). Returns the list of new ids."""
    k = len(picks)
    old_emb = model.emb.weight.data.clone()
    old_head = model.head_new.data.clone() if model.n_new else torch.zeros(0, model.d)
    old_bias = model.head_bias.data.clone() if model.n_new else torch.zeros(model.base_vocab)
    model.alloc_new_vocab(model.n_new + k)
    model.emb.weight.data[:old_emb.shape[0]] = old_emb
    model.head_new.data[:old_head.shape[0]] = old_head
    model.head_bias.data[:old_bias.shape[0]] = old_bias
    new_ids = []
    for a, b, n, q in picks:
        t = gt.add(a, b)
        model.emb.weight.data[t] = 0.5 * (model.emb.weight.data[a] + model.emb.weight.data[b])
        out_a = model.emb.weight.data[a] if a < model.base_vocab else model.head_new.data[a - model.base_vocab]
        model.head_new.data[t - model.base_vocab] = out_a
        ba = model.head_bias.data[a].item()
        model.head_bias.data[t] = ba + math.log(q)
        model.head_bias.data[a] = ba + math.log(1 - q)
        new_ids.append(t)
    model.nbase = gt.nbase_tensor()
    return new_ids


def rebuild_optimizer_resized(model, old_opt, lr, wd, old_params):
    """Like rebuild_optimizer, but carries Adam moments into params whose first dim grew (zero-padded rows)."""
    opt = rebuild_optimizer(model, old_opt, lr, wd)
    if old_opt is None:
        return opt
    names_new = dict(model.named_parameters())
    for name, old_p in old_params.items():
        st = old_opt.state.get(old_p)
        p = names_new.get(name)
        if not st or p is None or p is old_p:
            continue
        ns = {}
        for kk, v in st.items():
            if torch.is_tensor(v) and v.dim() >= 1 and v.shape != p.shape:
                z = torch.zeros_like(p.data); z[:v.shape[0]] = v; ns[kk] = z
            else:
                ns[kk] = v.clone() if torch.is_tensor(v) else v
        opt.state[p] = ns
    return opt


def merged_rows(arr, starts, T, gt):
    """Rows of T+1 ids from a base-token array, after applying gt's merges (identical to plain slicing when no merges)."""
    rows = []
    for s in starts:
        if not gt.merges:
            rows.append(np.asarray(arr[s:s + T + 1], dtype=np.int64))
            continue
        span = T + 1
        while True:
            ids = gt.apply(np.asarray(arr[s:s + span], dtype=np.int64).tolist())
            if len(ids) >= T + 1 or s + span >= len(arr):
                break
            span *= 2
        ids = (ids + [0] * (T + 1))[:T + 1]
        rows.append(np.asarray(ids, dtype=np.int64))
    a = torch.from_numpy(np.stack(rows)).to(DEV)
    return a[:, :-1].contiguous(), a[:, 1:].contiguous()


def ts_val_batches(valid, ctx, gt, n=12, B=16, seed=999):
    """Same draws as the original fixed TinyStories val (seed 999, 12x16); merges applied when the vocab grew."""
    r = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        s = r.integers(0, len(valid.a) - ctx - 1, size=B)
        out.append(merged_rows(valid.a, s, ctx, gt))
    return out

# --------------------------------------------------------------------------- v3: hunger, tree shape, ramps

def hunger_state(model, a):
    """RAM headroom under --mem-cap-mb. used = max(current RSS, params*16 bytes (weights+grad+Adam) + runtime estimate)."""
    rss, _ = rss_mb()
    est = model.n_params() * 16 / 2**20 + a.hunger_runtime_mb
    if DEV.type == "cuda":  # GPU build: weights/activations live in VRAM, host RSS is mostly the CUDA runtime
        used = max(torch.cuda.max_memory_reserved() / 2**20, est)
        free = 1.0 - used / a.vram_cap_mb
        return ("well_fed" if free > a.hunger_fed else ("tight" if free > a.hunger_tight else "near_cap")), free, used, rss
    used = max(rss if rss == rss else 0.0, est)
    free = 1.0 - used / a.mem_cap_mb
    st = "well_fed" if free > a.hunger_fed else ("tight" if free > a.hunger_tight else "near_cap")
    return st, free, used, rss


def target_width(li, a):
    """Roots (first --root-layers) aim for --root-width cells; branches taper by --taper per layer above the roots."""
    if li < a.root_layers:
        return a.root_width
    return max(a.branch_min_width, int(round(a.root_width * a.taper ** (li - a.root_layers + 1))))


def update_ramps(model, step):
    for layer in model.layers:
        for c in layer:
            if getattr(c, "ramp_born", None) is not None:
                r = min(1.0, max(0.0, (step - c.ramp_born) / max(1, c.ramp_steps)))
                c.ramp = r
                if r >= 1.0:
                    c.ramp, c.ramp_born = 1.0, None


def widths(model):
    return [len(l) for l in model.layers]


def remove_layer_if_empty(model, li):
    """Drop an empty layer (no cells => it computes nothing, so this is exact) and shift organ layer indices."""
    if len(model.layers[li]) > 0:
        return False
    del model.layers[li]
    for o in model.organs:
        if o["layer"] > li:
            o["layer"] -= 1
    return True

# --------------------------------------------------------------------------- v3: organizers / immune

@torch.no_grad()
def cell_activity(model, batches):
    """Mean output norm per (layer, cell) over batches (how 'busy' each cell looks)."""
    model.eval()
    acc = {}
    for xb, _ in batches:
        tr = []; model(xb, trace=tr)
        for li, t in enumerate(tr):
            for ci in range(t.shape[0]):
                acc[(li, ci)] = acc.get((li, ci), 0.0) + t[ci].mean().item() / len(batches)
    model.train()
    return acc


def plan_organizer_move(model, u, a):
    """Pick the least useful unprotected, non-organ cell sitting in a branch (or an over-full layer) and the
    most under-filled root to move it to. Returns (src_li, src_ci, dst_li, reason) or None."""
    widths_ = widths(model)
    deficits = [(target_width(li, a) - w) / target_width(li, a) for li, w in enumerate(widths_)]
    dsts = [(deficits[li], -li, li) for li in range(min(a.root_layers, len(model.layers))) if deficits[li] > 0]
    if not dsts:
        dsts = [(deficits[li], -li, li) for li in range(len(model.layers)) if deficits[li] > 0]
    if not dsts:
        return None
    _, _, dst = max(dsts)
    in_organ = {(o["layer"], m) for o in model.organs for m in o["members"]}
    cands = []
    for (li, ci), val in u.items():
        if li == dst or li < a.root_layers and deficits[li] >= 0:
            continue  # never take cells out of an unfilled root
        c = model.layers[li][ci]
        if cell_protected(model, li, c) or (li, c.cid) in in_organ or getattr(c, "ramp_born", None) is not None:
            continue
        over = widths_[li] > target_width(li, a)
        if over or li >= a.root_layers:
            cands.append((val, li, ci, over))
    if not cands:
        return None
    val, li, ci, over = min(cands)
    why = f"{model.layers[li][ci].cid} usefulness {val:.4f} in L{li} (width {widths_[li]}, target {target_width(li, a)}{', over-full' if over else ''}) -> root L{dst} (width {widths_[dst]}, target {target_width(dst, a)})"
    return li, ci, dst, why


@torch.no_grad()
def move_cell(model, li, ci, dst, step, ramp_steps):
    """Move a cell to layer dst; it fades in there 0->1 (so arriving is a no-op; leaving costs its usefulness).
    An emptied layer is removed. Returns the destination layer index after any removal."""
    cell = model.layers[li][ci]
    del model.layers[li][ci]
    cell.ramp, cell.ramp_born, cell.ramp_steps = 0.0, step, ramp_steps
    model.layers[dst].append(cell)
    removed = remove_layer_if_empty(model, li)
    return (dst - 1 if removed and dst > li else dst), removed


def find_freeloaders(model, u, act, base, a, step):
    """Cells that look busy (output norm >= median) but barely matter when switched off (impact < --immune-rel x loss)."""
    if not act:
        return []
    med = float(np.median(list(act.values())))
    out = []
    for (li, ci), val in u.items():
        c = model.layers[li][ci]
        if cell_protected(model, li, c) or step - c.born < a.prune_min_age or getattr(c, "ramp_born", None) is not None:
            continue
        ac = act.get((li, ci), 0.0)
        if ac >= med and val < a.immune_rel * base:
            out.append(dict(cell=c.cid, layer=li, ci=ci, impact=round(val, 5), activity=round(ac, 4),
                            ratio=round(val / max(ac, 1e-9), 6)))
    out.sort(key=lambda d: d["ratio"])
    return out


@torch.no_grad()
def reinit_cell(model, li, ci, step, ramp_steps):
    """Fresh weights (template init) for a freeloader; it fades in from 0, so the change == removing its old output."""
    c = model.layers[li][ci]
    fresh = Cell(model.d, **model.cell_kw)
    c.load_state_dict(fresh.state_dict())
    c.ramp, c.ramp_born, c.ramp_steps = 0.0, step, ramp_steps


# --------------------------------------------------------------------------- main loop

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="prepared token folder (from prep_data.py)")
    ap.add_argument("--data-dir", default=None, help="train on YOUR files: a folder of .txt/.md/.pdf (prepared automatically)")
    ap.add_argument("--out", default="runs/demo1")
    ap.add_argument("--minutes", type=float, default=45.0)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--anneal-steps", type=int, default=0, help="decay lr from --lr to 10% of it over this many steps from resume; independent of --minutes")
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=150)
    ap.add_argument("--plateau-window", type=int, default=600, help="steps to look back for improvement")
    ap.add_argument("--plateau-rel", type=float, default=0.012, help="grow if val improved less than this fraction over the window")
    ap.add_argument("--cooldown", type=int, default=600, help="min steps between growth events")
    ap.add_argument("--max-cells-per-layer", type=int, default=4)
    ap.add_argument("--max-layers", type=int, default=4)
    ap.add_argument("--width-before-depth", type=int, default=3)
    ap.add_argument("--depth-gain-rel", type=float, default=0.006, help="add depth if the last duplication bought less than this")
    ap.add_argument("--prune-rel", type=float, default=0.002, help="prune if switching a cell off costs less than this fraction of loss")
    ap.add_argument("--prune-min-age", type=int, default=1500)
    ap.add_argument("--prune-layer-min-age", type=int, default=3000, help="never prune inside a layer younger than this (new layers need time to switch on)")
    ap.add_argument("--fixed-shape", default=None, help="baseline mode: no growth, start at this shape, e.g. 4,4,4,1 = cells per layer")
    ap.add_argument("--max-params", type=float, default=20e6)
    ap.add_argument("--mem-cap-mb", type=float, default=1500)
    ap.add_argument("--stop-growing-frac", type=float, default=0.8, help="no growth in the last 20%% of the run (LR cool-down)")
    ap.add_argument("--prompts", default="Once upon a time|Lily wanted to|The little dog|Tom and his mom went to the",
                    help="'|'-separated prompts for the final text samples")
    ap.add_argument("--priority", default="idle", choices=["idle", "below_normal"])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--mix", action="append", default=None, help="share:path of a raw UTF-8 text file; repeat. Train on this weighted mix (streamed) instead of train.bin. val (TinyStories valid.bin) stays; val_mix is added")
    ap.add_argument("--mix-train-only", action="append", default=None, help="share:path trained on like --mix but NOT part of val_mix (keeps val_mix comparable when adding data)")
    ap.add_argument("--rotate-dir", default=None, help="folder of staged text chunks (*.txt) trained on like --mix-train-only; rotating data: the trainer re-scans it every --rotate-every batches without restarting (see rotation_stager.py)")
    ap.add_argument("--rotate-share", type=float, default=0.0, help="total mix share (same units as --mix) split evenly across the chunks in --rotate-dir")
    ap.add_argument("--rotate-every", type=int, default=200, help="re-scan --rotate-dir every N training batches")
    ap.add_argument("--mix-restart-tag", default="", help="when this tag differs from the checkpoint's, restart the --mix-ramp LR ramp and --mix-organ-grace at the resume step (once per tag)")
    ap.add_argument("--mix-holdout", type=float, default=0.01, help="tail fraction of every mix file reserved for val_mix (never trained on)")
    ap.add_argument("--mix-val-batches", type=int, default=12)
    ap.add_argument("--mix-ramp", type=int, default=1500, help="after switching to the mix, ramp LR from 0.2x to 1x over this many steps (softens the distribution shift)")
    ap.add_argument("--mix-organ-grace", type=int, default=4000, help="no organ dissolution for this many steps after switching to the mix")
    ap.add_argument("--decide-on", default="mix", choices=["mix", "ts"], help="with --mix: which val set drives plateau/prune/grow decisions (organ checks always use TinyStories val)")
    ap.add_argument("--organs", action="store_true", help="enable organ formation/specialization (groups of co-activating cells with a token-niche router)")
    ap.add_argument("--organ-start", type=int, default=1500, help="first step at which organs may form")
    ap.add_argument("--organ-every", type=int, default=450, help="steps between organ checks (rounded up to an eval point)")
    ap.add_argument("--organ-tau", type=float, default=0.5, help="min co-activation (corr of cell output norms across tokens) to group cells")
    ap.add_argument("--organ-lift-min", type=float, default=0.01, help="min loss rise (nats) on the niche when the organ is switched off")
    ap.add_argument("--organ-ratio", type=float, default=1.5, help="niche lift must be >= this x lift on other tokens")
    ap.add_argument("--organ-band", type=float, default=0.3, help="fraction of tokens per difficulty band (band 0 = hardest tokens)")
    ap.add_argument("--organ-aux", type=float, default=0.02, help="weight of the router specialization loss")
    ap.add_argument("--organ-max-per-layer", type=int, default=3)
    ap.add_argument("--organ-min-age", type=int, default=1500, help="min steps before an organ may dissolve")
    ap.add_argument("--organ-dissolve-frac", type=float, default=0.35, help="dissolve when niche lift falls below this fraction of its formation lift")
    # ---- v3 features (all default off)
    ap.add_argument("--wall-minutes", type=float, default=0, help="stop (and save) after this many minutes of THIS session (0 = off)")
    ap.add_argument("--trial-steps", type=int, default=150, help="training steps inside a structural trial before the keep/rollback check")
    ap.add_argument("--trial-tol", type=float, default=0.003, help="keep a trial only if decision-val loss rose by <= this (nats)")
    ap.add_argument("--hunger", action="store_true", help="1) hunger/scarcity rule from RAM headroom under --mem-cap-mb")
    ap.add_argument("--hunger-fed", type=float, default=0.50, help="free fraction above which Seed is well fed")
    ap.add_argument("--hunger-tight", type=float, default=0.15, help="free fraction above which Seed is tight (below = near cap)")
    ap.add_argument("--hunger-runtime-mb", type=float, default=300, help="RAM estimate for activations/runtime/data on top of params*16 bytes")
    ap.add_argument("--hunger-clear-gain", type=float, default=0.003, help="tight: only grow if the last growth bought at least this relative gain")
    ap.add_argument("--hunger-prune-tight", type=float, default=2.0, help="tight: prune threshold multiplier")
    ap.add_argument("--hunger-prune-near", type=float, default=3.0, help="near cap: prune threshold multiplier")
    ap.add_argument("--hunger-explore", type=float, default=0.5, help="well fed: organizer/immune/vocab intervals multiplied by this (more trials)")
    ap.add_argument("--tree-shape", action="store_true", help="2) roots & branches: prefer widening roots, taper width upward")
    ap.add_argument("--root-layers", type=int, default=5)
    ap.add_argument("--root-width", type=int, default=12, help="target cells per root layer")
    ap.add_argument("--taper", type=float, default=0.7, help="branch target width = root_width * taper^(layers above the roots)")
    ap.add_argument("--branch-min-width", type=int, default=1)
    ap.add_argument("--tree-root-fill", type=float, default=0.75, help="roots count as filled at this fraction of target width")
    ap.add_argument("--tree-depth-strict", type=float, default=0.5, help="while roots are unfilled, depth needs last split gain < this x --depth-gain-rel (soft preference)")
    ap.add_argument("--organizers", action="store_true", help="3) organizers: trial-move a weak branch cell into an under-filled root (rollback if worse)")
    ap.add_argument("--org-every", type=int, default=1500)
    ap.add_argument("--immune", action="store_true", help="4) white blood cells: ablation scan for busy-but-useless cells, trial prune/reinit")
    ap.add_argument("--immune-every", type=int, default=2000)
    ap.add_argument("--immune-rel", type=float, default=0.001, help="freeloader if switching it off costs < this x loss while its output norm >= median")
    ap.add_argument("--grow-vocab", action="store_true", help="5) growing tokenizer: add merged tokens from recent text (mass-split, function preserving)")
    ap.add_argument("--vocab-every", type=int, default=3000)
    ap.add_argument("--vocab-k", type=int, default=16, help="merges per growth event")
    ap.add_argument("--vocab-max", type=int, default=4096, help="max extra tokens on top of the base vocab")
    ap.add_argument("--vocab-min-count", type=int, default=200, help="min pair count in the recent window")
    ap.add_argument("--vocab-window", type=int, default=200, help="recent training batches kept for pair counting")
    ap.add_argument("--protect", default="", help="6) comma-separated cell/organ ids to tag protected at start (tags in the checkpoint are always honored)")
    ap.add_argument("--exact-split", action="store_true", help="duplicate with antisymmetric output noise: exactly function preserving (old split perturbs the child's inputs/gate)")
    ap.add_argument("--msg-ramp", action="store_true", help="MSG-style: new layers keep the template weights and fade in 0->1 instead of zeroed outputs")
    ap.add_argument("--ramp-steps", type=int, default=300, help="fade-in steps for ramped cells (MSG layers, organizer moves, immune reinit)")
    ap.add_argument("--device", default="cpu", help="GPU build: cpu or cuda")
    ap.add_argument("--init-parts", nargs="+", default=None, help="fresh start only: base cell + these part files (export_part.py .safetensors), each PATH[@LAYER] (default layer 0); imported protected, fading in from 0")
    ap.add_argument("--vram-cap-mb", type=float, default=5000, help="GPU build: growth stops before peak reserved VRAM would pass this")
    a = ap.parse_args()
    global DEV
    DEV = torch.device("cuda" if a.device == "cuda" and torch.cuda.is_available() else "cpu")
    if DEV.type == "cuda":
        torch.set_default_device(DEV)
    print(f"device: {DEV}" + (f" ({torch.cuda.get_device_name(0)}), VRAM cap {a.vram_cap_mb:.0f} MB" if DEV.type == "cuda" else ""), flush=True)

    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    pri = lower_priority(a.priority)
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    data = Path(a.data)
    if a.data_dir:
        import hashlib
        src = Path(a.data_dir).resolve()
        data = Path("data_custom") / f"{src.name}_{hashlib.sha1(str(src).encode()).hexdigest()[:8]}"
        if not (data / "meta.json").exists():
            from prep_data import build_from_folder
            print(f"Preparing your files from {src} -> {data} ...", flush=True)
            build_from_folder(src, data)
        else:
            print(f"Using prepared data {data} (delete or rename it to re-read the folder)", flush=True)
    meta = json.loads((data / "meta.json").read_text())
    train, valid = Tokens(data / "train.bin"), Tokens(data / "valid.bin")
    from tokenizers import Tokenizer
    base_tok = Tokenizer.from_file(str(Path(meta["tokenizer"]) if Path(meta["tokenizer"]).exists() else data / Path(meta["tokenizer"]).name))
    ck = out / "latest.pt"
    c_resume = torch.load(ck, weights_only=False, map_location=DEV) if (a.resume and ck.exists()) else None
    gt = GrowTok(base_tok, (c_resume or {}).get("merges"))
    tok = gt  # same interface (.encode().ids / .decode / .token_to_id); identical ids while no merges exist

    val_batches = ts_val_batches(valid, a.ctx, gt)  # == the original fixed val when the vocab never grew
    use_batches = val_batches[:4]
    probe_x = val_batches[0][0][:4]
    mix = None; dec_batches = val_batches
    if a.mix:
        mix = TextMix(a.mix, tok, tok.token_to_id("<eos>"), a.mix_holdout, train_only=a.mix_train_only, rotate_dir=a.rotate_dir, rotate_share=a.rotate_share, rotate_every=a.rotate_every)
        t_v = time.time()
        mix_val = mix.val_batches(a.mix_val_batches, 16, a.ctx)
        print("mix: " + ", ".join(f"{s['name']} {p*100:.1f}% ({s['size']/2**20:.0f} MB)" for s, p in zip(mix.src, mix.p))
              + f" | val_mix {a.mix_val_batches}x16 held-out rows built in {time.time()-t_v:.1f}s", flush=True)
        if a.decide_on == "mix":
            dec_batches = mix_val; use_batches = mix_val[:4]
    mix_p = out / "metrics_mix.csv" if a.mix else None

    cell_kw = dict(n_head=2, head_dim=32, mlp=256)
    step, t_offset = 0, 0.0
    state = dict(last_growth=0, last_action=None, val_at_last_dup=None, events=0, last_organ_check=0, next_oid=1)
    if a.resume and ck.exists():
        c = c_resume
        model = SeedLM.from_structure(meta["vocab"], a.d, a.ctx, cell_kw, c["structure"], c["next_id"])
        model.set_organs(c.get("organs", []))
        if c.get("n_new", 0):
            model.alloc_new_vocab(c["n_new"]); model.nbase = gt.nbase_tensor()
            assert model.vocab == gt.vocab_size(), "checkpoint vocab and merges disagree"
        model.load_state_dict(c["model"])
        opt = rebuild_optimizer(model, None, a.lr, a.wd)
        try:
            if c.get("opt") is None:
                raise ValueError("no optimizer state (parts were imported)")
            opt.load_state_dict(c["opt"])
        except Exception as e:
            print(f"optimizer state not loaded ({e}); starting fresh Adam moments", flush=True)
        step, t_offset, state = c["step"], c["elapsed"], c["state"]
        print(f"resumed at step {step}")
        if a.mix and a.decide_on == "mix" and state.get("decide_on") != "mix":
            state["val_at_last_dup"] = None  # old value was on TinyStories val, not comparable
            state["mix_start_step"] = step
        if a.mix: state["decide_on"] = a.decide_on
        if a.mix and "mix_start_step" not in state: state["mix_start_step"] = step
    elif a.fixed_shape:
        shape = [int(n) for n in a.fixed_shape.split(",")]
        struct, k = [], 0
        for n in shape:
            struct.append([dict(cid=f"c{k + i}", parent=None, born=0) for i in range(n)]); k += n
        torch.manual_seed(a.seed)
        model = SeedLM.from_structure(meta["vocab"], a.d, a.ctx, cell_kw, struct, k)
        opt = rebuild_optimizer(model, None, a.lr, a.wd)
    else:
        model = SeedLM(meta["vocab"], a.d, a.ctx, cell_kw)
        opt = rebuild_optimizer(model, None, a.lr, a.wd)
        if a.init_parts:  # optional parts: fresh base cell + imported part(s), each function-preserving (fades in from 0)
            from import_part import import_part_into
            v0 = evaluate(model, val_batches)
            c0 = dict(model=model.state_dict(), opt=None, structure=model.structure(), next_id=model.next_id, organs=[],
                      step=0, state=state, args=vars(a), vocab=model.vocab, base_vocab=model.base_vocab, n_new=0,
                      base_tokenizer_sha=base_tok_sha(base_tok))
            for spec in a.init_parts:
                ppath, _, lay = spec.partition("@")
                cids, li, pm = import_part_into(c0, ppath, int(lay) if lay else 0, a.ramp_steps, protect=True)
                print(f"init-part {pm['kind']} {pm['id']} ({pm.get('n_cells', len(cids))} cells, from {pm.get('source', '?')} step {pm.get('step')}) -> L{li} as {cids}, protected", flush=True)
            model = SeedLM.from_structure(meta["vocab"], a.d, a.ctx, cell_kw, c0["structure"], c0["next_id"])
            model.set_organs(c0["organs"]); model.load_state_dict(c0["model"]); state = c0["state"]
            opt = rebuild_optimizer(model, None, a.lr, a.wd)
            print(f"init-parts: val_ts base cell {v0:.4f} -> with parts {evaluate(model, val_batches):.4f} (equal = function-preserving)", flush=True)

    if a.mix and a.mix_restart_tag and state.get("mix_tag") != a.mix_restart_tag:
        state["mix_start_step"] = step; state["mix_tag"] = a.mix_restart_tag
        print(f"mix restart '{a.mix_restart_tag}' at step {step}: LR ramp {a.mix_ramp} steps, organ grace {a.mix_organ_grace} steps", flush=True)
    if a.mix and "mix_start_step" in state:
        a.organ_grace_until = state["mix_start_step"] + a.mix_organ_grace
    model.organ_aux_w = a.organ_aux if a.organs else 0.0; model.organ_band = a.organ_band
    c_resume = None
    for pid in [s.strip() for s in a.protect.split(",") if s.strip()]:
        hit = False
        for layer in model.layers:
            for cc in layer:
                if cc.cid == pid: cc.protected = True; hit = True
        for o in model.organs:
            if o["oid"] == pid: o["protected"] = True; hit = True
        print(f"protect {pid}: {'tagged' if hit else 'NOT FOUND'}", flush=True)
    ev_f = open(out / "events.jsonl", "a", encoding="utf-8")
    met_p = out / "metrics.csv"
    new_csv = not met_p.exists()
    met_f = open(met_p, "a", newline="", encoding="utf-8"); met = csv.writer(met_f)
    if new_csv:
        met.writerow(["step", "minutes", "train_loss", "val_loss", "layers", "cells", "params", "tok_s", "rss_mb", "lr", "sibling_sim", "organs"])
    log_f = open(out / "train.log", "a", encoding="utf-8")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True); log_f.write(line + "\n"); log_f.flush()

    def event(kind, **kw):
        rec = dict(kind=kind, step=step, minutes=round((time.time() - t0 + t_offset) / 60, 2),
                   layers=len(model.layers), cells=model.n_cells(), params=model.n_params(),
                   shape=model.shape_str(), **kw)
        ev_f.write(json.dumps(rec) + "\n"); ev_f.flush()
        log(f"EVENT {kind}: " + json.dumps({k: v for k, v in rec.items() if k not in ('kind',)}))
        state["events"] += 1

    def sample(prompt, n=60):
        ids = torch.tensor([tok.encode(prompt).ids])
        model.eval(); outp = model.generate(ids, n); model.train()
        return tok.decode(outp[0].tolist()).replace("<eos>", " / ")

    def save(tag=None):
        c = dict(model=model.state_dict(), opt=opt.state_dict(), structure=model.structure(),
                 next_id=model.next_id, organs=model.organs, step=step, elapsed=time.time() - t0 + t_offset, state=state,
                 args=vars(a), vocab=model.vocab, base_vocab=model.base_vocab, n_new=model.n_new,
                 merges=[[ma, mb] for ma, mb, _ in gt.merges], base_tokenizer_sha=base_tok_sha(base_tok),
                 tokenizer_version=f"{base_tok_sha(base_tok)}+{len(gt.merges)}m:{gt.sha()}", seed_version="v3")
        if gt.merges:
            (out / "tokenizer_grown.json").write_text(json.dumps(dict(
                base_tokenizer=str(meta["tokenizer"]), base_sha=base_tok_sha(base_tok), base_vocab=gt.base,
                vocab=gt.vocab_size(), merges_sha=gt.sha(), note="ids >= base are merges [a, b] applied in this order after base BPE",
                merges=[[ma, mb, mn, gt.decode([mn])] for ma, mb, mn in gt.merges]), ensure_ascii=False, indent=0), encoding="utf-8")
        torch.save(c, out / "latest.tmp"); os.replace(out / "latest.tmp", ck)
        if tag:
            (out / "snapshots").mkdir(exist_ok=True)
            torch.save(c, out / "snapshots" / f"{tag}.pt")

    t0 = time.time()
    budget = a.minutes * 60
    log(f"Seed start: mix={'on decide_on=' + a.decide_on if a.mix else 'off'} threads={a.threads} low_priority={pri} vocab={meta['vocab']} params={model.n_params():,} shape={model.shape_str()}")
    if step == 0:
        event("birth", val_loss=round(evaluate(model, val_batches), 4))
    hist = []  # (step, val)
    ema = None; tok_count = 0; t_last = time.time()
    from collections import deque
    pair_buf = deque(maxlen=a.vocab_window)
    v3_p = out / "v3_metrics.csv"
    hstate = [None]

    def get_batch():
        if mix:
            return mix.batch(a.batch, a.ctx, rng)
        if gt.merges:
            s = rng.integers(0, len(train.a) - a.ctx - 1, size=a.batch)
            return merged_rows(train.a, s, a.ctx, gt)
        return train.batch(a.batch, a.ctx, rng)

    def train_step(lr_):
        for g in opt.param_groups:
            g["lr"] = lr_
        update_ramps(model, step)
        x, y = get_batch()
        if a.grow_vocab:
            pair_buf.append(x.cpu().numpy().copy())
        loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        return loss.item(), x.numel()

    def run_trial(kind, apply_fn, why, lr_):
        """Snapshot -> apply -> train --trial-steps -> keep only if decision-val rose by <= --trial-tol, else roll back."""
        nonlocal model, opt, step
        snap_m, snap_o, snap_s, snap_step = copy.deepcopy(model), copy.deepcopy(opt.state_dict()), copy.deepcopy(state), step
        before = evaluate(model, dec_batches)
        info = apply_fn() or {}
        opt = rebuild_optimizer(model, opt, lr_, a.wd)
        for p in info.pop("_reset_params", []):
            opt.state.pop(p, None)
        right_after = evaluate(model, dec_batches)
        for _ in range(a.trial_steps):
            train_step(lr_); step += 1
        after = evaluate(model, dec_batches)
        keep = after <= before + a.trial_tol
        if not keep:
            model = snap_m
            opt = rebuild_optimizer(model, None, lr_, a.wd); opt.load_state_dict(snap_o)
            state.clear(); state.update(snap_s); step = snap_step
        event(f"{kind}_{'keep' if keep else 'rollback'}", why=why, val_before=round(before, 4), val_right_after=round(right_after, 4),
              val_after_trial=round(after, 4), trial_steps=a.trial_steps, tol=a.trial_tol, **info)
        return keep

    @torch.no_grad()
    def probs(xb):
        model.eval(); p = F.softmax(model(xb), -1); model.train(); return p

    def grow_vocab_now(lr_):
        nonlocal opt, val_batches, mix_val, dec_batches, use_batches
        tot, left = count_pairs(list(pair_buf), tok.token_to_id("<eos>"), model.vocab)
        room = a.vocab_max - model.n_new
        tot = {kk: n for kk, n in tot.items() if (kk // model.vocab, kk % model.vocab) not in gt.rank}  # never re-add a merge
        picks = pick_merges(tot, left, model.vocab, min(a.vocab_k, room), a.vocab_min_count)
        pair_buf.clear()  # old rows were encoded with the old merges; count afresh next time
        if not picks:
            log("vocab: no pair frequent enough"); return False
        bt, bm = evaluate(model, val_batches), (evaluate(model, mix_val) if mix else None)
        xb = val_batches[0][0][:4]
        p0 = probs(xb); v_old = model.vocab
        old_params = dict(model.named_parameters())
        new_ids = expand_vocab(model, gt, picks)
        p1 = probs(xb)
        pb = p1[..., :v_old].clone()
        for (pa, _, _, _), t in zip(picks, new_ids):
            pb[..., pa] += p1[..., t]
        err = (pb - p0).abs().max().item()
        opt = rebuild_optimizer_resized(model, opt, lr_, a.wd, old_params)
        bt_same = evaluate(model, val_batches)  # same base-token batches, new model
        val_batches = ts_val_batches(valid, a.ctx, gt)
        if mix:
            mix_val = mix.val_batches(a.mix_val_batches, 16, a.ctx)
        dec_batches = mix_val if (mix and a.decide_on == "mix") else val_batches
        use_batches = dec_batches[:4]
        hist.clear(); state["val_at_last_dup"] = None; state["val_at_growth"] = None
        at, am = evaluate(model, val_batches), (evaluate(model, mix_val) if mix else None)
        event("vocab_grow", added=len(new_ids), vocab=model.vocab,
              merges=[dict(id=t, a=pa, b=pb_, text=gt.decode([t]), count=n, q=round(q, 4)) for (pa, pb_, n, q), t in zip(picks, new_ids)],
              mass_split_max_abs_err=err, val_ts_before=round(bt, 4), val_ts_same_batches_after=round(bt_same, 4),
              val_ts_reencoded_after=round(at, 4), val_mix_before=None if bm is None else round(bm, 4),
              val_mix_reencoded_after=None if am is None else round(am, 4), note="val = nats per base token")
        return True

    model.train()
    if a.anneal_steps and state.get("anneal_steps_param") != a.anneal_steps:
        state["anneal_start_step"] = step
        state["anneal_steps_param"] = a.anneal_steps
    state.setdefault("anneal_start_step", step)
    while True:
        elapsed = time.time() - t0 + t_offset
        if elapsed >= budget:
            break
        if a.wall_minutes and time.time() - t0 >= a.wall_minutes * 60:
            break
        frac = elapsed / budget
        lr = a.lr * min(1.0, (step + 1) / a.warmup)
        if a.mix and "mix_start_step" in state and a.mix_ramp > 0:
            lr *= 0.2 + 0.8 * min(1.0, (step - state["mix_start_step"]) / a.mix_ramp)
        if frac > a.stop_growing_frac:  # cool down LR over the last part of the run
            lr *= max(0.1, 1 - (frac - a.stop_growing_frac) / (1 - a.stop_growing_frac))
        if a.anneal_steps:
            lr *= max(0.1, 1 - 0.9 * min(1.0, max(0.0, (step - state["anneal_start_step"]) / a.anneal_steps)))
        lv, nt = train_step(lr)
        step += 1; tok_count += nt
        ema = lv if ema is None else 0.98 * ema + 0.02 * lv

        if step % a.eval_every == 0:
            v = evaluate(model, val_batches)
            vm = evaluate(model, mix_val) if mix else None
            vd = vm if (mix and a.decide_on == "mix") else v
            now = time.time(); tps = tok_count / (now - t_last); tok_count = 0; t_last = now
            rss, peak = rss_mb()
            sim = sibling_similarity(model, probe_x)
            met.writerow([step, round((now - t0 + t_offset) / 60, 3), round(ema, 4), round(v, 4), len(model.layers),
                          model.n_cells(), model.n_params(), round(tps), round(rss), f"{lr:.2e}",
                          "" if sim is None else round(sim, 3), len(model.organs)])
            met_f.flush()
            if mix:
                new_m = not mix_p.exists()
                with open(mix_p, "a", newline="", encoding="utf-8") as mf:
                    w = csv.writer(mf)
                    if new_m: w.writerow(["step", "minutes", "train_loss", "val_ts", "val_mix", "decide_on", "cells", "organs", "tok_s", "rss_mb"])
                    w.writerow([step, round((now - t0 + t_offset) / 60, 3), round(ema, 4), round(v, 4), round(vm, 4), a.decide_on,
                                model.n_cells(), len(model.organs), round(tps), round(rss)])
            vms = f" val_mix {vm:.3f}" if mix else ""
            hun, hfree, hused, _ = hunger_state(model, a)
            if a.hunger and hun != hstate[0]:
                if hstate[0] is not None:
                    event("hunger_change", old=hstate[0], new=hun, free_frac=round(hfree, 3), used_mb=round(hused), cap_mb=a.mem_cap_mb)
                hstate[0] = hun
            hs = f" | hunger {hun} {hfree*100:.0f}% free" if a.hunger else ""
            log(f"step {step} train {ema:.3f} val {v:.3f}{vms} | {model.shape_str()} | {model.n_params()/1e6:.2f}M params | {tps:,.0f} tok/s | RSS {rss:.0f} MB VRAM {(torch.cuda.max_memory_reserved()/2**20 if DEV.type == 'cuda' else 0):.0f} MB | sib_sim {sim} | vocab {model.vocab}{hs}")
            new_v3 = not v3_p.exists()
            with open(v3_p, "a", newline="", encoding="utf-8") as vf:
                w = csv.writer(vf)
                if new_v3: w.writerow(["step", "minutes", "train_loss", "val_ts", "val_mix", "layers", "cells", "organs", "vocab", "hunger", "free_frac", "rss_mb", "widths"])
                w.writerow([step, round((now - t0 + t_offset) / 60, 3), round(ema, 4), round(v, 4), "" if vm is None else round(vm, 4),
                            len(model.layers), model.n_cells(), len(model.organs), model.vocab, hun if a.hunger else "", round(hfree, 3), round(rss),
                            " ".join(map(str, widths(model)))])
            hist.append((step, vd))
            grow_ok = not a.fixed_shape and frac < a.stop_growing_frac
            explore = a.hunger_explore if (a.hunger and hun == "well_fed") else 1.0
            maint = False

            # ---- organs
            if a.organs and step >= a.organ_start and step - state.get("last_organ_check", 0) >= a.organ_every:
                state["last_organ_check"] = step
                if organ_check(model, val_batches, tok, a, step, state, out, event, log):
                    opt = rebuild_optimizer(model, opt, lr, a.wd)
                    log("sample: " + sample(a.prompts.split("|")[0]))
            # ---- v3 maintenance (at most one per eval point)
            vstart = state.get("mix_start_step", 0)
            if (grow_ok and a.grow_vocab and (not a.hunger or hun == "well_fed") and model.n_new < a.vocab_max
                    and step - state.get("last_vocab", vstart) >= a.vocab_every * explore and len(pair_buf) >= min(50, a.vocab_window)):
                state["last_vocab"] = step; maint = True
                if grow_vocab_now(lr):
                    save(tag=f"step{step}_vocab{model.vocab}")
            if not maint and grow_ok and a.immune and step - state.get("last_immune", vstart) >= a.immune_every * explore:
                state["last_immune"] = step; maint = True
                ib, iu = usefulness(model, use_batches)
                act = cell_activity(model, use_batches)
                fl = find_freeloaders(model, iu, act, ib, a, step)
                event("immune_scan", n_flagged=len(fl), flagged=fl[:12], median_activity=round(float(np.median(list(act.values()))), 4) if act else None)
                if fl:
                    f0 = fl[0]; li, ci = f0["layer"], f0["ci"]
                    cid = model.layers[li][ci].cid
                    in_org = any(o["layer"] == li and cid in o["members"] for o in model.organs)
                    if len(model.layers[li]) > 1 and not in_org:
                        def _prune(li=li, ci=ci, cid=cid):
                            path = prune_cell(model, li, ci, out / "archive", step, f"immune freeloader impact {f0['impact']}")
                            return dict(action="prune", cell=cid, layer=li, archived=str(path), impact=f0["impact"], activity=f0["activity"])
                        run_trial("immune", _prune, f"{cid} busy (norm {f0['activity']}) but impact {f0['impact']}", lr)
                    else:
                        def _reinit(li=li, ci=ci, cid=cid):
                            reinit_cell(model, li, ci, step, a.ramp_steps)
                            return dict(action="reinit", cell=cid, layer=li, impact=f0["impact"], activity=f0["activity"],
                                        _reset_params=list(model.layers[li][ci].parameters()))
                        run_trial("immune", _reinit, f"{cid} busy (norm {f0['activity']}) but impact {f0['impact']}", lr)
                    save()
            if not maint and grow_ok and a.organizers and step - state.get("last_org", vstart) >= a.org_every * explore:
                state["last_org"] = step; maint = True
                ob, ou = usefulness(model, use_batches)
                plan = plan_organizer_move(model, ou, a)
                if plan is None:
                    log("organizer: nothing worth moving")
                else:
                    sli, sci, dli, why = plan
                    def _move(sli=sli, sci=sci, dli=dli):
                        cid = model.layers[sli][sci].cid
                        nd, removed = move_cell(model, sli, sci, dli, step, a.ramp_steps)
                        return dict(cell=cid, from_layer=sli, to_layer=nd, removed_empty_layer=removed, src_usefulness=round(ou[(sli, sci)], 5))
                    run_trial("organizer_move", _move, why, lr)
                    save()
            # ---- growth decision
            since = step - state["last_growth"]
            past = [h for h in hist if h[0] <= step - a.plateau_window]
            can_grow = not maint and grow_ok and since >= a.cooldown and past and past[-1][0] >= state["last_growth"]
            if can_grow:
                rel_gain = (past[-1][1] - vd) / past[-1][1]
                if rel_gain < a.plateau_rel:
                    base, u = usefulness(model, use_batches)
                    ustr = {model.layers[li][ci].cid: round(val, 4) for (li, ci), val in u.items()}
                    log(f"plateau: val gain {rel_gain*100:.2f}% over {a.plateau_window} steps; usefulness {ustr}")
                    acted = False
                    pmult = 1.0
                    if a.hunger:
                        pmult = {"well_fed": 1.0, "tight": a.hunger_prune_tight, "near_cap": a.hunger_prune_near}[hun]
                    growth_gain = None
                    if state.get("val_at_growth"):
                        growth_gain = (state["val_at_growth"] - vd) / state["val_at_growth"]
                    # 1) prune dead cells (never the last cell of a layer, never protected parts)
                    for (li, ci), val in sorted(u.items(), key=lambda kv: (-kv[0][0], -kv[0][1])):
                        cell = model.layers[li][ci]
                        layer_age = step - min(c.born for c in model.layers[li])
                        if cell_protected(model, li, cell) or getattr(cell, "ramp_born", None) is not None:
                            continue
                        if (val < a.prune_rel * pmult * base and step - cell.born >= a.prune_min_age and len(model.layers[li]) > 1
                                and layer_age >= a.prune_layer_min_age):
                            cid = cell.cid
                            before = evaluate(model, dec_batches)  # same batches as val_after
                            path = prune_cell(model, li, ci, out / "archive", step, f"usefulness {val:.5f}")
                            organ_on_prune(model, cid, li, out / "archive", step, event)
                            opt = rebuild_optimizer(model, opt, lr, a.wd)
                            event("prune", cell=cid, layer=li, usefulness=round(val, 5), archived=str(path),
                                  val_before=round(before, 4), val_after=round(evaluate(model, dec_batches), 4))
                            acted = True
                    if acted:
                        base, u = usefulness(model, use_batches)
                    # 2) grow
                    rss, peak = rss_mb()
                    # leave headroom for one more cell (weights+grads+Adam ~16 bytes/param) under the RAM cap
                    cell_mb = sum(p.numel() for p in model.layers[0][0].parameters()) * 16 / 2**20
                    room = model.n_params() < a.max_params and ((max(rss, peak) + cell_mb < a.mem_cap_mb) if DEV.type != 'cuda' else (rss + 150 < a.mem_cap_mb and torch.cuda.max_memory_reserved() / 2**20 * (1 + 1.0 / max(1, model.n_cells())) + cell_mb < a.vram_cap_mb))
                    if not room:
                        log(f"growth blocked: params {model.n_params():,} RSS {rss:.0f} MB peak {peak:.0f} MB cap {a.mem_cap_mb} MB" + (f" | VRAM peak {torch.cuda.max_memory_reserved()/2**20:.0f}/{a.vram_cap_mb:.0f} MB" if DEV.type == 'cuda' else ""))
                    if room and a.hunger and hun == "near_cap":
                        room = False; log(f"hunger near_cap ({hfree*100:.0f}% free): no growth, reorganize only")
                    if room and a.hunger and hun == "tight" and growth_gain is not None and growth_gain < a.hunger_clear_gain:
                        room = False; log(f"hunger tight: last growth bought {growth_gain*100:.2f}% < {a.hunger_clear_gain*100:.2f}%, not growing")
                    top = len(model.layers) - 1
                    last_dup_gain = None
                    if state["last_action"] == "duplicate" and state["val_at_last_dup"]:
                        last_dup_gain = (state["val_at_last_dup"] - vd) / state["val_at_last_dup"]
                    top_wide = len(model.layers[top]) >= a.width_before_depth
                    dup_small = last_dup_gain is not None and last_dup_gain < a.depth_gain_rel
                    want_depth = len(model.layers) < a.max_layers and (top_wide or dup_small)
                    reasons = [f"top layer width {len(model.layers[top])}{'>=' if top_wide else '<'}{a.width_before_depth}",
                               "last split gain " + ("n/a" if last_dup_gain is None else f"{last_dup_gain*100:.2f}%")]
                    if a.hunger and hun == "well_fed" and want_depth and not (top_wide and (last_dup_gain is None or dup_small)):
                        want_depth = False; reasons.append("well fed: widen first (depth needs a wide top AND a small last split)")
                    roots_filled = None
                    if a.tree_shape:
                        rw = widths(model)[:a.root_layers]
                        roots_filled = len(rw) >= a.root_layers and all(w >= a.tree_root_fill * target_width(i, a) for i, w in enumerate(rw))
                        strict = last_dup_gain is not None and last_dup_gain < a.depth_gain_rel * a.tree_depth_strict
                        if len(model.layers) < a.root_layers:
                            want_depth = want_depth or all(w >= a.tree_root_fill * target_width(i, a) for i, w in enumerate(rw))
                        elif want_depth and not roots_filled and not strict:
                            want_depth = False; reasons.append(f"roots unfilled {rw} vs target {a.root_width}: widen (depth only if split gain < {a.depth_gain_rel*a.tree_depth_strict*100:.2f}%)")
                        elif want_depth:
                            reasons.append("roots filled" if roots_filled else "split gains collapsed: depth allowed despite unfilled roots (soft rule)")
                    if room and (a.tree_shape or a.hunger):
                        event("shape_decision", choice="depth" if want_depth else "width", reasons=reasons, widths=widths(model),
                              roots_filled=roots_filled, hunger=hun if a.hunger else None)
                    if room and want_depth:
                        best_ci = max(range(len(model.layers[top])), key=lambda ci: u.get((top, ci), 0))
                        before = evaluate(model, dec_batches)
                        cell = add_layer(model, step, model.layers[top][best_ci], a.ramp_steps if a.msg_ramp else 0)
                        opt = rebuild_optimizer(model, opt, lr, a.wd)
                        event("add_layer", new_cell=cell.cid, template=cell.parent, reason=(
                            "top layer full" if len(model.layers[top]) >= a.width_before_depth else f"last split gained only {last_dup_gain*100:.2f}%"),
                              val_before=round(before, 4), val_after=round(evaluate(model, dec_batches), 4))
                        state["last_action"] = "add_layer"; acted = True
                    elif room:
                        cands = [(val, li, ci) for (li, ci), val in u.items() if len(model.layers[li]) < a.max_cells_per_layer]
                        if a.tree_shape:  # favour layers below their target width, roots most (soft: weights, not a ban)
                            tw = [(val * (1 + (2.0 if li < a.root_layers else 1.0) * max(0, target_width(li, a) - len(model.layers[li])) / target_width(li, a))
                                   * (1.0 if len(model.layers[li]) < target_width(li, a) else 0.25), li, ci) for val, li, ci in cands]
                            cands = tw
                        if cands:
                            val_u, li, ci = max(cands)
                            before = evaluate(model, dec_batches)
                            child, pmap = duplicate_cell(model, li, ci, step, exact=a.exact_split)
                            opt = rebuild_optimizer(model, opt, lr, a.wd, pmap)
                            event("duplicate", parent=child.parent, child=child.cid, layer=li,
                                  parent_usefulness=round(val_u, 4), val_before=round(before, 4),
                                  val_after=round(evaluate(model, dec_batches), 4))
                            state["last_action"] = "duplicate"; state["val_at_last_dup"] = vd; acted = True
                    if acted:
                        state["last_growth"] = step; state["val_at_growth"] = vd
                        save(tag=f"step{step}_{len(model.layers)}L_{model.n_cells()}c")
                        log("sample: " + sample(a.prompts.split("|")[0]))
        if step % 1000 == 0:
            save()

    v = evaluate(model, val_batches)
    rss, peak = rss_mb()
    save(tag=f"final_step{step}")
    base, u = usefulness(model, use_batches)
    samples = {p: sample(p, 80) for p in a.prompts.split("|") if p.strip()}
    summary = dict(step=step, minutes=round((time.time() - t0 + t_offset) / 60, 1), val_loss=round(v, 4),
                   layers=len(model.layers), cells=model.n_cells(), params=model.n_params(), shape=model.shape_str(),
                   rss_mb=round(rss), peak_rss_mb=round(peak), threads=a.threads,
                   usefulness={model.layers[li][ci].cid: round(val, 4) for (li, ci), val in u.items()},
                   organs=model.organs, samples=samples)
    event("final", val_loss=round(v, 4))
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log("FINAL " + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
