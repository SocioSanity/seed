"""Seed v4 FINAL: a growing, symmetric, hot-swappable language model (trainer + model + parts format).

Design in one paragraph: the model is a pre-norm GQA transformer whose width is a whole number of ORGANS.
An organ is one column: exactly one cell in every layer (a cell = 4 query heads + a SwiGLU slice; the layer
shares K/V heads, norms, QK-norm gains, RoPE). Every layer always has the same organs (symmetric). Each layer
has a router: tokens are split between organs by softmax competition, with a load-balance loss and a sibling
decorrelation penalty against redundancy. Growth = widen (+1 organ) or deepen (+2 layers spread through the
stack), and each growth is accepted only if it beats a real no-growth control trained the same steps.
Parts (cell / organ / tree) are safetensors + a JSON manifest, can be exported any time, and can be hot-installed
or hot-removed during training through an inbox folder, with exact function-preserving fade-in/fade-out.
An organizer keeps organs coherent (function-preserving column alignment), an immune system replaces
freeloader cells (trial vs control), and hunger reads RAM/VRAM headroom to decide how bold growth can be.
See SEED_FINAL_NOTES.md and PARTS_README.md.  seed_v3_gpu.py is imported read-only for the data pipeline.
"""
import argparse, copy, csv, hashlib, itertools, json, math, os, queue, shutil, sys, threading, time, uuid
os.environ.setdefault("RAYON_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ARCH = "seed-v4-final"
PART_FORMAT = "seed-part/1"
DEV = torch.device("cpu")

# --------------------------------------------------------------------------- standard tokenizer


TOK_MANIFEST = "seed_tokenizer_manifest.json"


def file_sha(path, n=16):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:n]


def ensure_std_tokenizer(tok_dir, src_json, name="seed-bpe4096", version="1"):
    """The shared, versioned tokenizer every Seed, part and chat uses: <dir>/tokenizer.json + seed_tokenizer_manifest.json
    (not TOKENIZER.json: Windows file names are case-insensitive, it would overwrite tokenizer.json)."""
    tok_dir = Path(tok_dir)
    man_p = tok_dir / TOK_MANIFEST
    if man_p.exists():
        return json.loads(man_p.read_text(encoding="utf-8"))
    tok_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_json, tok_dir / "tokenizer.json")
    from tokenizers import Tokenizer
    t = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))
    man = dict(name=name, version=version, id=f"{name}-v{version}", sha256=file_sha(tok_dir / "tokenizer.json"),
               vocab_size=t.get_vocab_size(), eos="<eos>", eos_id=t.token_to_id("<eos>"),
               note="Byte-level BPE trained on TinyStoriesV2 (prep_data.py). Never edit: a new tokenizer = a new version.",
               created=time.strftime("%Y-%m-%d %H:%M:%S"))
    man_p.write_text(json.dumps(man, indent=1), encoding="utf-8")
    return man


def load_std_tokenizer(tok_dir):
    from tokenizers import Tokenizer
    tok_dir = Path(tok_dir)
    man = json.loads((tok_dir / TOK_MANIFEST).read_text(encoding="utf-8"))
    if file_sha(tok_dir / "tokenizer.json") != man["sha256"]:
        raise SystemExit(f"tokenizer.json in {tok_dir} does not match its manifest sha (it was edited?)")
    return Tokenizer.from_file(str(tok_dir / "tokenizer.json")), man

# --------------------------------------------------------------------------- model


def rope_freqs(T, hd, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
    return torch.polar(torch.ones(T, hd // 2), torch.outer(torch.arange(T, dtype=torch.float32), inv))


def apply_rope(x, fc):
    """Interleaved RoPE as one complex multiply (fewer memory passes than rotate-half). x: (B, T, ..., hd)."""
    T = x.shape[1]
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    f = fc[:T].view(1, T, *([1] * (x.dim() - 3)), -1)
    return torch.view_as_real(xc * f).flatten(-2).type_as(x)


SOCKET_VERSION = "seed-socket-v1"
SOCKET_D = 384          # canonical residual width at the socket
ROPE_BASE = 10000.0     # canonical positional convention (interleaved complex RoPE)
SOCKET_HEAD_DIM = 32    # RoPE pairs are defined per 32-dim head in v1
ADAPTER_KEYS = ("ain", "bin", "aout", "bout")


class Cell(nn.Module):
    """One organ's slice of one layer: n_head query heads + one SwiGLU slice. alpha = fade level (0 = absent).
    socket=True (seed-socket-v1): the cell owns its K/V head and q/k norms, and has low-rank in/out adapters
    (input  u = h + bin(ain h),  output o' = o + bout(aout o); bin = bout = 0 => identity), so the cell is
    self-contained and portable between seeds."""

    def __init__(self, d, n_head=4, head_dim=32, mlp=192, zero_out=False, out_std=0.02, socket=False, rank=8):
        super().__init__()
        self.q = nn.Linear(d, n_head * head_dim, bias=False)
        self.proj = nn.Linear(n_head * head_dim, d, bias=False)
        self.w12 = nn.Linear(d, 2 * mlp, bias=False)
        self.down = nn.Linear(mlp, d, bias=False)
        self.register_buffer("alpha", torch.ones(()))
        nn.init.normal_(self.q.weight, std=d ** -0.5)
        nn.init.normal_(self.w12.weight, std=d ** -0.5)
        for w in (self.proj.weight, self.down.weight):
            nn.init.zeros_(w) if zero_out else nn.init.normal_(w, std=out_std)
        self.socket = bool(socket)
        if socket:
            self.kv = nn.Linear(d, 2 * head_dim, bias=False)       # one K/V head per cell (MQA inside the cell)
            nn.init.normal_(self.kv.weight, std=d ** -0.5)
            self.q_norm = nn.Parameter(torch.ones(head_dim))
            self.k_norm = nn.Parameter(torch.ones(head_dim))
            self.ain = nn.Parameter(torch.randn(rank, d) * d ** -0.5)
            self.bin = nn.Parameter(torch.zeros(d, rank))
            self.aout = nn.Parameter(torch.randn(rank, d) * d ** -0.5)
            self.bout = nn.Parameter(torch.zeros(d, rank))
        self.cid, self.protected, self.adapt = "c0", False, False
        self.plugin = False   # plug-in organ cell: own sigmoid gate, outside the host cells' softmax competition


CELL_KEYS = ("q", "proj", "w12", "down")


def cell_state(c):
    """name -> tensor for everything a cell owns (the unit that moves between seeds)."""
    t = {k: getattr(c, k).weight for k in CELL_KEYS}
    if c.socket:
        t["kv"] = c.kv.weight
        for k in ("q_norm", "k_norm") + ADAPTER_KEYS:
            t[k] = getattr(c, k)
    return t


@torch.no_grad()
def load_cell_state(c, t, prefix=""):
    for k, v in cell_state(c).items():
        if prefix + k in t:
            v.copy_(t[prefix + k].to(v.device, v.dtype))


class Layer(nn.Module):
    def __init__(self, d, K, ckw, zero_out=False, out_std=0.02):
        super().__init__()
        self.d, self.ckw = d, ckw
        nh, hd = ckw["n_head"], ckw["head_dim"]
        self.socket = bool(ckw.get("socket"))
        self.n1 = nn.Parameter(torch.ones(d), requires_grad=not self.socket)   # socket: canonical norm, gain fixed at 1
        self.n2 = nn.Parameter(torch.ones(d), requires_grad=not self.socket)
        if not self.socket:
            self.kv = nn.Linear(d, 2 * nh * hd, bias=False)
            nn.init.normal_(self.kv.weight, std=d ** -0.5)
            self.q_norm = nn.Parameter(torch.ones(hd))
            self.k_norm = nn.Parameter(torch.ones(hd))
        self.cells = nn.ModuleList([Cell(d, **ckw, zero_out=zero_out, out_std=out_std) for _ in range(K)])
        self.router = nn.Parameter(torch.zeros(K, d))
        self.capture = False
        self.last_p = None

    def forward(self, x, fc, mask=None, want_aux=False, sub=16):
        if self.socket:
            return self.forward_socket(x, fc, mask, want_aux, sub)
        B, T, d = x.shape
        K = len(self.cells)
        nh, hd, m = self.ckw["n_head"], self.ckw["head_dim"], self.ckw["mlp"]
        c_ = nh * hd
        cs = self.cells
        al = torch.stack([c.alpha for c in cs])                       # (K,) fade levels
        h = F.rms_norm(x, (d,), self.n1)
        qkv = F.linear(h, torch.cat([c.q.weight for c in cs] + [self.kv.weight]))  # one matmul: all q + shared kv
        q = qkv[..., :K * c_].reshape(B, T, K, nh, hd)
        kv = qkv[..., K * c_:].reshape(B, T, 2, nh, hd)
        q = apply_rope(F.rms_norm(q, (hd,), self.q_norm), fc)
        k = apply_rope(F.rms_norm(kv[:, :, 0], (hd,), self.k_norm), fc)
        v = kv[:, :, 1]
        q = q.reshape(B, T, K * nh, hd).transpose(1, 2)
        k = k.unsqueeze(2).expand(B, T, K, nh, hd).reshape(B, T, K * nh, hd).transpose(1, 2)
        v = v.unsqueeze(2).expand(B, T, K, nh, hd).reshape(B, T, K * nh, hd).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, K, c_)
        eff = al if mask is None else al * mask
        x = x + F.linear((a * eff.view(1, 1, K, 1).to(a.dtype)).reshape(B, T, K * c_),
                         torch.cat([c.proj.weight for c in cs], dim=1))
        h2 = F.rms_norm(x, (d,), self.n2)
        u = F.linear(h2, torch.cat([c.w12.weight for c in cs])).view(B, T, K, 2, m)
        act = F.silu(u[:, :, :, 0]) * u[:, :, :, 1]
        g, p = self._gate(h2, al, mask, B, T, K)
        x = x + F.linear((act * g.to(act.dtype).unsqueeze(-1)).reshape(B, T, K * m),
                         torch.cat([c.down.weight for c in cs], dim=1))
        return self._aux(x, a, act, p, cs, want_aux, sub, K)

    def _gate(self, h2, al, mask, B, T, K):
        p = None
        plug = [c.plugin for c in self.cells]
        if any(plug):
            pm = torch.tensor(plug, device=h2.device)
            z = F.linear(h2, self.router).float()
            zl = z + torch.log(al.clamp_min(1e-30)).masked_fill(al == 0, float("-inf"))
            p = torch.softmax(zl.masked_fill(pm, float("-inf")), dim=-1)          # host cells compete as before
            g = al.masked_fill(pm, 0).sum() * p + (al * torch.sigmoid(z)).masked_fill(~pm, 0)
            if mask is not None:
                g = g * mask
            if self.capture:
                self.last_p = (g / g.sum(-1, keepdim=True).clamp_min(1e-6)).detach()
            self._host_p = p[..., ~pm]
            return g, p
        self._host_p = None
        if K > 1:   # competition: p_c = alpha_c e^z_c / sum alpha_j e^z_j ; gate = (sum alpha) * p  (exact at alpha=0)
            z = F.linear(h2, self.router).float()
            z = z + torch.log(al.clamp_min(1e-30)).masked_fill(al == 0, float("-inf"))
            p = torch.softmax(z, dim=-1)
            g = al.sum() * p
        else:
            g = al.view(1, 1, 1).expand(B, T, 1)
        if mask is not None:
            g = g * mask
        if self.capture and p is not None:
            self.last_p = p.detach()
        return g, p

    def _aux(self, x, a, act, p, cs, want_aux, sub, K):
        if not want_aux or K == 1:
            z0 = x.new_zeros(()).float()
            return x, z0, z0
        att_s = torch.einsum("btkc,kdc->btkd", a[:, ::sub], torch.stack([c.proj.weight for c in cs]))
        mo_s = torch.einsum("btkm,kdm->btkd", act[:, ::sub], torch.stack([c.down.weight for c in cs]))
        dec = 0.5 * (decor(att_s) + decor(mo_s))
        hp = p if self._host_p is None else self._host_p
        Kh = hp.shape[-1]
        if Kh < 2:
            return x, dec, x.new_zeros(()).float()
        mp = hp.mean(dim=(0, 1))
        return x, dec, Kh * (mp * mp).sum() - 1.0

    def forward_socket(self, x, fc, mask, want_aux, sub):
        B, T, d = x.shape
        K = len(self.cells)
        nh, hd, m = self.ckw["n_head"], self.ckw["head_dim"], self.ckw["mlp"]
        c_, o_ = nh * hd, (nh + 2) * hd
        cs = self.cells
        adapt = any(c.adapt for c in cs)
        al = torch.stack([c.alpha for c in cs])
        h = F.rms_norm(x, (d,), self.n1)
        Wqkv = [torch.cat([c.q.weight, c.kv.weight]) for c in cs]
        qkv = F.linear(h, torch.cat(Wqkv)).view(B, T, K, o_)
        if adapt:   # u = h + bin(ain h): low-rank, folded into each input matrix (W u = W h + (W bin)(ain h))
            A = torch.cat([c.ain for c in cs])
            zin = F.linear(h, A).view(B, T, K, -1)
            qkv = qkv + torch.einsum("btkr,kor->btko", zin, torch.stack([W @ c.bin for W, c in zip(Wqkv, cs)]))
        qkv = qkv.view(B, T, K, nh + 2, hd)
        qn = torch.stack([c.q_norm for c in cs]).view(1, 1, K, 1, hd)
        kn = torch.stack([c.k_norm for c in cs]).view(1, 1, K, hd)
        q = apply_rope(F.rms_norm(qkv[:, :, :, :nh], (hd,)) * qn, fc)
        k = apply_rope(F.rms_norm(qkv[:, :, :, nh], (hd,)) * kn, fc)
        v = qkv[:, :, :, nh + 1]
        q = q.reshape(B, T, K * nh, hd).transpose(1, 2)
        k = k.unsqueeze(3).expand(B, T, K, nh, hd).reshape(B, T, K * nh, hd).transpose(1, 2)
        v = v.unsqueeze(3).expand(B, T, K, nh, hd).reshape(B, T, K * nh, hd).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, K, c_)
        eff = al if mask is None else al * mask
        if any(c.plugin for c in cs):   # plug-in cells gate their attention write with sigmoid(router . h)
            pm = torch.tensor([c.plugin for c in cs], device=x.device)
            sg = torch.where(pm, torch.sigmoid(F.linear(h, self.router).float()), torch.ones((), device=x.device))
            ag = a * (eff.view(1, 1, K) * sg).unsqueeze(-1).to(a.dtype)
        else:
            ag = a * eff.view(1, 1, K, 1).to(a.dtype)
        x = x + F.linear(ag.reshape(B, T, K * c_), torch.cat([c.proj.weight for c in cs], dim=1))
        if adapt:   # o' = o + bout(aout o)
            Bo = torch.cat([c.bout for c in cs], dim=1)
            zo = torch.einsum("btkc,krc->btkr", ag, torch.stack([c.aout @ c.proj.weight for c in cs]))
            x = x + F.linear(zo.reshape(B, T, -1), Bo)
        h2 = F.rms_norm(x, (d,), self.n2)
        u = F.linear(h2, torch.cat([c.w12.weight for c in cs])).view(B, T, K, 2 * m)
        if adapt:
            zin2 = F.linear(h2, A).view(B, T, K, -1)
            u = u + torch.einsum("btkr,kor->btko", zin2, torch.stack([c.w12.weight @ c.bin for c in cs]))
        u = u.view(B, T, K, 2, m)
        act = F.silu(u[:, :, :, 0]) * u[:, :, :, 1]
        g, p = self._gate(h2, al, mask, B, T, K)
        am = act * g.to(act.dtype).unsqueeze(-1)
        x = x + F.linear(am.reshape(B, T, K * m), torch.cat([c.down.weight for c in cs], dim=1))
        if adapt:
            zo2 = torch.einsum("btkm,krm->btkr", am, torch.stack([c.aout @ c.down.weight for c in cs]))
            x = x + F.linear(zo2.reshape(B, T, -1), Bo)
        return self._aux(x, a, act, p, cs, want_aux, sub, K)


def decor(o):
    """Mean squared cosine similarity between sibling cell outputs (0 = orthogonal, 1 = identical)."""
    z = o.float().reshape(-1, o.shape[2], o.shape[3])
    z = z / (z.norm(dim=-1, keepdim=True) + 1e-6)
    G = torch.bmm(z, z.transpose(1, 2))
    K = G.shape[1]
    off = G.pow(2).sum(dim=(1, 2)) - G.diagonal(dim1=1, dim2=2).pow(2).sum(-1)
    return off.mean() / (K * (K - 1))


class SeedV4(nn.Module):
    def __init__(self, vocab, d, ctx, L, K, ckw, tie=True, softcap=30.0):
        super().__init__()
        self.vocab, self.d, self.ctx, self.ckw, self.tie, self.softcap = vocab, d, ctx, dict(ckw), tie, softcap
        self.emb = nn.Embedding(vocab, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        out_std = 0.02 / math.sqrt(2 * max(1, L))
        self.layers = nn.ModuleList([Layer(d, K, self.ckw, out_std=out_std) for _ in range(L)])
        self.nf = nn.Parameter(torch.ones(d))
        if not tie:
            self.head = nn.Parameter(torch.randn(vocab, d) * 0.02)
        self.register_buffer("fc", rope_freqs(ctx, ckw["head_dim"], ROPE_BASE), persistent=False)
        self.ckpt_layers = 0
        self.organ_ids = [f"o{i+1}" for i in range(K)]
        self.protected_organs = set()
        self.next_cell, self.next_organ = 1, K + 1
        self.lineage = uuid.uuid4().hex[:12]
        self.tokenizer = {}
        self.socket = SOCKET_VERSION if ckw.get("socket") else None
        self.anchor = {}          # socket anchor id/sha this seed is tied to
        self.organ_meta = {}      # organ id -> {name, version, sha256, source} for adopted organs
        if self.socket:          # canonical anchor: frozen shared (un)embedding, final norm gain fixed at 1
            self.emb.weight.requires_grad_(False)
            self.nf.requires_grad_(False)
        for layer in self.layers:
            for c in layer.cells:
                c.cid = self.new_cid()

    def new_cid(self):
        i = self.next_cell; self.next_cell += 1; return f"c{i}"

    def new_oid(self):
        i = self.next_organ; self.next_organ += 1; return f"o{i}"

    @property
    def L(self):
        return len(self.layers)

    @property
    def K(self):
        return len(self.layers[0].cells)

    def shape_str(self):
        return f"{self.L}x{self.K}"

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def cells_iter(self):
        for li, layer in enumerate(self.layers):
            for ci, c in enumerate(layer.cells):
                yield li, ci, c

    def find_cell(self, cid):
        for li, ci, c in self.cells_iter():
            if c.cid == cid:
                return li, ci
        return None

    def structure(self):
        return dict(arch=ARCH, vocab=self.vocab, d=self.d, ctx=self.ctx, ckw=self.ckw, tie=self.tie, softcap=self.softcap,
                    L=self.L, K=self.K, organ_ids=list(self.organ_ids), protected_organs=sorted(self.protected_organs),
                    cids=[[c.cid for c in l.cells] for l in self.layers],
                    protected_cells=[c.cid for _, _, c in self.cells_iter() if c.protected],
                    adapt_cells=[c.cid for _, _, c in self.cells_iter() if c.adapt], organ_meta=self.organ_meta,
                    plugin_cells=[c.cid for _, _, c in self.cells_iter() if c.plugin],
                    socket=self.socket, anchor=self.anchor,
                    next_cell=self.next_cell, next_organ=self.next_organ, lineage=self.lineage, tokenizer=self.tokenizer)

    @staticmethod
    def from_structure(st):
        m = SeedV4(st["vocab"], st["d"], st["ctx"], st["L"], st["K"], st["ckw"], tie=st["tie"], softcap=st["softcap"])
        m.organ_ids = list(st["organ_ids"]); m.protected_organs = set(st.get("protected_organs", []))
        pc = set(st.get("protected_cells", [])); ac = set(st.get("adapt_cells", [])); plc = set(st.get("plugin_cells", []))
        for layer, ids in zip(m.layers, st["cids"]):
            for c, cid in zip(layer.cells, ids):
                c.cid, c.protected, c.adapt, c.plugin = cid, cid in pc, cid in ac, cid in plc
        m.organ_meta = dict(st.get("organ_meta", {})); m.anchor = dict(st.get("anchor", {}))
        m.next_cell, m.next_organ, m.lineage = st["next_cell"], st["next_organ"], st["lineage"]
        m.tokenizer = st.get("tokenizer", {})
        return m

    def forward(self, idx, targets=None, skip=None, stats=None, per_token=False):
        x = self.emb(idx)
        want = stats is not None
        for li, layer in enumerate(self.layers):
            mask = None
            if skip:
                ks = [ci for (l2, ci) in skip if l2 == li]
                if ks:
                    mask = torch.ones(len(layer.cells), device=x.device)
                    mask[ks] = 0.0
            if li < self.ckpt_layers and self.training and torch.is_grad_enabled():
                x, dl, bl = torch.utils.checkpoint.checkpoint(layer, x, self.fc, mask, want, use_reentrant=False)
            else:
                x, dl, bl = layer(x, self.fc, mask, want)
            if want and len(layer.cells) > 1:
                stats["decor"].append(dl); stats["bal"].append(bl)
        x = F.rms_norm(x, (self.d,), self.nf)
        W = self.emb.weight if self.tie else self.head
        logits = F.linear(x, W.to(x.dtype)).float()
        if self.softcap:
            logits = self.softcap * torch.tanh(logits / self.softcap)
        if targets is None:
            return logits
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1),
                               reduction="none" if per_token else "mean")

    @torch.no_grad()
    def generate(self, idx, n, temp=0.8, top_k=40, top_p=0.95, rep_pen=1.1, stop_id=None, prior="auto"):
        pr = self.__dict__.get("decode_prior") if isinstance(prior, str) else prior
        for _ in range(n):
            logits = self(idx[:, -self.ctx:])[:, -1, :].float()
            if pr is not None:   # brace-bug fix: re-split the collapsed-anchor token group by real frequencies
                logits[0] += prior_bias(pr, int(idx[0, -1]), logits.size(-1), logits.device)
            if rep_pen and rep_pen != 1.0:
                seen = idx[0, -128:].unique()
                lg = logits[0, seen]
                logits[0, seen] = torch.where(lg > 0, lg / rep_pen, lg * rep_pen)
            logits = logits / max(temp, 1e-5)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, -1)
            if top_p and top_p < 1.0:
                sp, si = torch.sort(probs, descending=True)
                cut = sp.cumsum(-1) - sp > top_p
                sp[cut] = 0
                probs = torch.zeros_like(probs).scatter_(-1, si, sp)
                probs = probs / probs.sum(-1, keepdim=True)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], 1)
            if stop_id is not None and int(nxt) == stop_id:
                break
        return idx

# --------------------------------------------------------------------------- structural ops (all symmetric)


@torch.no_grad()
def widen(model, oid=None):
    """+1 organ: one fresh cell in EVERY layer (zero outputs = no-op), router row = mean of the old rows."""
    oid = oid or model.new_oid()
    new = []
    for layer in model.layers:
        c = Cell(model.d, **model.ckw, zero_out=True).to(layer.n1.device)
        c.cid = model.new_cid()
        layer.cells.append(c); new.append(c)
        r = layer.router.data
        layer.router = nn.Parameter(torch.cat([r, r.mean(0, keepdim=True)], 0))
    model.organ_ids.append(oid)
    return new


@torch.no_grad()
def deepen(model, n):
    """+n full-width layers spread evenly; input side copied from the layer below, outputs zero (exact no-op)."""
    L = model.L
    pos = sorted({max(1, min(L, int(round(L * (j + 1) / n)))) for j in range(n)}, reverse=True)
    new = []
    for p in pos:
        src = model.layers[p - 1]
        lay = Layer(model.d, model.K, model.ckw, zero_out=True).to(src.n1.device)
        lay.n1.copy_(src.n1); lay.n2.copy_(src.n2); lay.router.copy_(src.router)
        if not lay.socket:
            lay.kv.weight.copy_(src.kv.weight); lay.q_norm.copy_(src.q_norm); lay.k_norm.copy_(src.k_norm)
        for cn, cs in zip(lay.cells, src.cells):
            cn.q.weight.copy_(cs.q.weight); cn.w12.weight.copy_(cs.w12.weight)
            if cn.socket:   # input side copied (stacking-style), outputs and adapters stay zero
                cn.kv.weight.copy_(cs.kv.weight); cn.q_norm.copy_(cs.q_norm); cn.k_norm.copy_(cs.k_norm)
            cn.cid = model.new_cid(); cn.alpha.copy_(cs.alpha)
        model.layers.insert(p, lay)
        new.append(lay)
    return new


def new_cell_ids(model, new_mods):
    ids = set()
    for li, layer in enumerate(model.layers):
        for ci, c in enumerate(layer.cells):
            if any(c is m for m in new_mods) or any(layer is m for m in new_mods):
                ids.add((li, ci))
    return ids


@torch.no_grad()
def remove_organ_column(model, k):
    """Physically remove column k from every layer (call only when its alpha is 0 => exact)."""
    gone = []
    for layer in model.layers:
        gone.append(layer.cells[k])
        keep = [i for i in range(len(layer.cells)) if i != k]
        layer.cells = nn.ModuleList([layer.cells[i] for i in keep])
        layer.router = nn.Parameter(layer.router.data[keep].clone())
    oid = model.organ_ids.pop(k)
    model.protected_organs.discard(oid)
    return oid, gone


@torch.no_grad()
def reinit_cell(model, li, ci):
    """Fresh cell in place: new random input weights, zero outputs, alpha 0 (exact no-op until faded in), router row = mean."""
    layer = model.layers[li]; c = layer.cells[ci]
    d = model.d
    nn.init.normal_(c.q.weight, std=d ** -0.5); nn.init.normal_(c.w12.weight, std=d ** -0.5)
    c.proj.weight.zero_(); c.down.weight.zero_(); c.alpha.fill_(0.0)   # caller fades it in
    if c.socket:
        nn.init.normal_(c.kv.weight, std=d ** -0.5); c.q_norm.fill_(1.0); c.k_norm.fill_(1.0)
        c.bin.zero_(); c.bout.zero_(); c.adapt = False
    if len(layer.cells) > 1:
        others = [i for i in range(len(layer.cells)) if i != ci]
        layer.router.data[ci] = layer.router.data[others].mean(0)
    c.cid, c.protected = model.new_cid(), False
    return list(c.parameters())

# --------------------------------------------------------------------------- optimisers


def zeropower_ns5_batched(G, steps=5):
    """Newton-Schulz orthogonalisation (Muon) for a stack of same-shape matrices. fp32 (Turing has no fast bf16)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    tr = X.size(-2) > X.size(-1)
    if tr:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        X = a * X + (b * A + c * A @ A) @ X
    return X.mT if tr else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            ps = [p for p in g["params"] if p.grad is not None]
            if not ps:
                continue
            bufs = []
            for p in ps:
                st = self.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)
                bufs.append(st["momentum_buffer"])
            grads = [p.grad.float() for p in ps]
            torch._foreach_mul_(bufs, g["momentum"]); torch._foreach_add_(bufs, grads)
            ups = torch._foreach_add(grads, bufs, alpha=g["momentum"]) if g["nesterov"] else bufs
            if g["weight_decay"]:
                torch._foreach_mul_(ps, 1 - g["lr"] * g["weight_decay"])
            byshape = {}
            for p, u in zip(ps, ups):
                byshape.setdefault(tuple(p.shape), []).append((p, u))
            for shp, items in byshape.items():
                O = zeropower_ns5_batched(torch.stack([u for _, u in items]))
                scale = -g["lr"] * max(1.0, shp[0] / shp[1]) ** 0.5
                for (p, _), o in zip(items, O):
                    p.add_(o.to(p.dtype), alpha=scale)


def split_params(model):
    muon, adam_decay, adam_nodecay = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if (".cells." in n or n.endswith(".kv.weight")) and p.dim() == 2 and not n.rsplit(".", 1)[-1] in ADAPTER_KEYS:
            muon.append(p)
        elif n in ("emb.weight", "head"):
            adam_decay.append(p)
        else:
            adam_nodecay.append(p)
    return muon, adam_decay, adam_nodecay


def build_opts(model, a, old=None, reset=()):
    """New optimisers; state of surviving params carried over by identity (params in `reset` start fresh)."""
    mu, ad, an = split_params(model)
    muon = Muon(mu, lr=a.muon_lr, momentum=0.95, weight_decay=a.muon_wd)
    adam = torch.optim.AdamW([{"params": ad, "weight_decay": a.wd}, {"params": an, "weight_decay": 0.0}],
                             lr=a.adam_lr, betas=(0.9, 0.95), eps=1e-8, fused=(DEV.type == "cuda"))
    for o, base in ((muon, a.muon_lr), (adam, a.adam_lr)):
        for g in o.param_groups:
            g["base_lr"] = base
    rs = set(id(p) for p in reset)
    if old is not None:
        for new_o, old_o in zip((muon, adam), old):
            for g in new_o.param_groups:
                for p in g["params"]:
                    st = old_o.state.get(p)
                    if st and id(p) not in rs:
                        new_o.state[p] = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in st.items()}
    return muon, adam

# --------------------------------------------------------------------------- EMA


class EMA:
    def __init__(self, model, decay):
        self.decay, self.n = decay, 0
        self.sh = {p: p.detach().float().clone() for p in model.parameters()}

    def sync(self, model, reset=()):
        rs = set(id(p) for p in reset)
        self.sh = {p: self.sh[p] if (p in self.sh and self.sh[p].shape == p.shape and id(p) not in rs)
                   else p.detach().float().clone() for p in model.parameters()}

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        ps = list(model.parameters())
        torch._foreach_lerp_([self.sh[p] for p in ps], [p.detach() for p in ps], 1 - d)

    @torch.no_grad()
    def swap(self, model):
        for p in model.parameters():
            t = p.detach().clone(); p.copy_(self.sh[p].to(p.dtype)); self.sh[p] = t.float()

    def state_by_name(self, model):
        return {n: self.sh[p] for n, p in model.named_parameters()}

    def load_by_name(self, model, sd):
        for n, p in model.named_parameters():
            if n in sd and sd[n].shape == p.shape:
                self.sh[p] = sd[n].to(p.device).float()

# --------------------------------------------------------------------------- saving that never crashes


def tensor_bytes(o):
    if torch.is_tensor(o):
        return o.numel() * o.element_size()
    if isinstance(o, dict):
        return sum(tensor_bytes(v) for v in o.values())
    if isinstance(o, (list, tuple)):
        return sum(tensor_bytes(v) for v in o)
    return 0


def safe_save(obj, path, log, fallback_dir=None, retries=4, margin_mb=500, writer=None):
    path = Path(path)
    need = int(tensor_bytes(obj) * 1.2) + margin_mb * 2**20
    targets = [path] + ([Path(fallback_dir) / path.name] if fallback_dir else [])
    for tgt in targets:
        for attempt in range(retries):
            tmp = tgt.with_suffix(tgt.suffix + ".tmp")
            try:
                tgt.parent.mkdir(parents=True, exist_ok=True)
                free = shutil.disk_usage(tgt.parent).free
                if free < need:
                    log(f"SAVE skip {tgt}: only {free/2**30:.2f} GB free, need {need/2**30:.2f} GB")
                    break
                (writer or torch.save)(obj, tmp)
                if tmp.stat().st_size < 64:
                    raise IOError("temp file suspiciously small")
                os.replace(tmp, tgt)
                return tgt
            except Exception as e:
                log(f"SAVE attempt {attempt+1}/{retries} to {tgt} failed: {type(e).__name__}: {e}")
                try:
                    if tmp.exists():
                        tmp.unlink()  # our own half-written temp file only
                except Exception:
                    pass
                time.sleep(min(30, 2 ** attempt))
    log("SAVE FAILED everywhere; training continues and will try again at the next save point")
    return None

# --------------------------------------------------------------------------- parts: safetensors + JSON manifest


def part_paths(stem):
    stem = Path(stem)
    return stem.with_suffix(".safetensors"), stem.with_suffix(".json")


def model_sig(model):
    return dict(d=model.d, n_head=model.ckw["n_head"], head_dim=model.ckw["head_dim"], mlp=model.ckw["mlp"],
                vocab=model.vocab, arch=ARCH, rope="interleaved", socket=model.socket)


def write_part(stem, tensors, manifest, log=print):
    from safetensors.torch import save_file
    st_p, js_p = part_paths(stem)
    st_p.parent.mkdir(parents=True, exist_ok=True)
    tensors = {k: v.detach().float().cpu().contiguous() for k, v in tensors.items()}
    ok = safe_save(tensors, st_p, log, margin_mb=50,
                   writer=lambda obj, p: save_file(obj, str(p), metadata={"format": PART_FORMAT, "kind": manifest["kind"]}))
    if not ok:
        return None
    manifest = dict(manifest, format=PART_FORMAT, file=st_p.name, sha256=file_sha(st_p),
                    n_params=int(sum(v.numel() for v in tensors.values())), created=time.strftime("%Y-%m-%d %H:%M:%S"))
    tmp = js_p.with_suffix(".json.tmp"); tmp.write_text(json.dumps(manifest, indent=1), encoding="utf-8"); os.replace(tmp, js_p)
    return js_p


def read_part(js_path):
    from safetensors.torch import load_file
    js_path = Path(js_path)
    man = json.loads(js_path.read_text(encoding="utf-8"))
    st_p = js_path.with_name(man.get("file", js_path.with_suffix(".safetensors").name))
    if man.get("format") != PART_FORMAT:
        raise ValueError(f"unknown part format {man.get('format')!r}")
    if file_sha(st_p) != man.get("sha256"):
        raise ValueError("safetensors file does not match the manifest sha256 (corrupt or edited)")
    return man, load_file(str(st_p))


def cell_tensors(layer, ci, prefix=""):
    c = layer.cells[ci]
    t = {prefix + k: v for k, v in cell_state(c).items()}
    t[prefix + "router"] = layer.router[ci]
    return t


def export_cell(model, li, ci, stem, extra=None, log=print):
    c = model.layers[li].cells[ci]
    man = dict(kind="cell", id=c.cid, layer=li, organ=model.organ_ids[ci], n_layers=model.L, protected=c.protected,
               sig=model_sig(model), tokenizer=model.tokenizer, lineage=model.lineage, **(extra or {}))
    return write_part(stem, cell_tensors(model.layers[li], ci), man, log)


def export_organ(model, k, stem, extra=None, log=print):
    t = {}
    for li, layer in enumerate(model.layers):
        t.update(cell_tensors(layer, k, prefix=f"L{li}."))
    oid = model.organ_ids[k]
    man = dict(kind="organ", id=oid, n_layers=model.L, cids=[l.cells[k].cid for l in model.layers],
               protected=oid in model.protected_organs, sig=model_sig(model), tokenizer=model.tokenizer,
               lineage=model.lineage, **(extra or {}))
    return write_part(stem, t, man, log)


def export_tree(model, stem, extra=None, log=print):
    t = {k: v for k, v in model.state_dict().items()}
    man = dict(kind="tree", id=f"tree-{model.lineage}", structure=model.structure(), sig=model_sig(model),
               tokenizer=model.tokenizer, lineage=model.lineage, **(extra or {}))
    return write_part(stem, t, man, log)


def load_tree(js_path, device="cpu"):
    man, t = read_part(js_path)
    if man["kind"] != "tree":
        raise ValueError("not a tree part")
    m = SeedV4.from_structure(man["structure"])
    t = {k: v.float() if getattr(v, "is_floating_point", lambda: False)() else v for k, v in t.items()}
    m.load_state_dict(t)
    return m.to(device), man


def validate_part(model, man, t):
    """Hard errors (refuse) and warnings for installing part `man` into `model`."""
    errs, warns = [], []
    sig = model_sig(model)
    for k, v in sig.items():
        if man.get("sig", {}).get(k) != v:
            errs.append(f"{k}: part {man.get('sig', {}).get(k)!r} != model {v!r}")
    mt, pt = model.tokenizer.get("sha256"), man.get("tokenizer", {}).get("sha256")
    if mt and pt and mt != pt:
        errs.append(f"tokenizer mismatch: part {man['tokenizer'].get('id')} vs model {model.tokenizer.get('id')}")
    if man["kind"] == "organ":
        if man.get("n_layers") != model.L:
            errs.append(f"organ has {man.get('n_layers')} layers, model has {model.L}")
        for li in range(min(model.L, man.get("n_layers", 0))):
            for key, ref in cell_state(model.layers[li].cells[0]).items():
                tt = t.get(f"L{li}.{key}")
                if tt is None or tuple(tt.shape) != tuple(ref.shape):
                    errs.append(f"L{li}.{key}: shape {None if tt is None else tuple(tt.shape)} != {tuple(ref.shape)}")
    elif man["kind"] == "cell":
        for key, ref in cell_state(model.layers[0].cells[0]).items():
            if key not in t or tuple(t[key].shape) != tuple(ref.shape):
                errs.append(f"{key}: bad shape")
    elif man["kind"] == "tree":
        errs.append("a tree is a whole model: start a run from it (--init-tree) or chat with it; it is not hot-installed")
    if man.get("lineage") and man["lineage"] != model.lineage:
        warns.append(f"part grew in another lineage ({man['lineage']}); it may need fine-tuning to help here")
    return errs, warns


@torch.no_grad()
def install_organ(model, man, t, oid=None):
    """Append the organ as a new column with alpha = 0 everywhere (exactly function preserving)."""
    oid = oid or man["id"]
    if oid in model.organ_ids:
        oid = model.new_oid()
    for li, layer in enumerate(model.layers):
        c = Cell(model.d, **model.ckw).to(layer.n1.device)
        load_cell_state(c, t, prefix=f"L{li}.")
        c.adapt = bool(any(float(t[f"L{li}.{k}"].abs().sum()) > 0 for k in ("bin", "bout") if f"L{li}.{k}" in t))
        c.alpha.zero_(); c.cid = model.new_cid(); c.protected = bool(man.get("protected"))
        layer.cells.append(c)
        layer.router = nn.Parameter(torch.cat([layer.router.data, t[f"L{li}.router"].to(layer.router.device).view(1, -1)], 0))
    model.organ_ids.append(oid)
    if man.get("protected"):
        model.protected_organs.add(oid)
    return oid


@torch.no_grad()
def load_cell_weights(model, li, ci, t, prefix=""):
    c = model.layers[li].cells[ci]
    load_cell_state(c, t, prefix)
    model.layers[li].router.data[ci] = t[prefix + "router"].to(model.layers[li].router.device)
    return list(c.parameters())

# --------------------------------------------------------------------------- eval / misc


def rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2**20
    except Exception:
        return float("nan")


def sys_avail_mb():
    try:
        import psutil
        return psutil.virtual_memory().available / 2**20
    except Exception:
        return float("inf")


@torch.no_grad()
def evaluate(model, batches, amp, skip=None):
    was = model.training
    model.eval()
    tot = 0.0
    for x, y in batches:
        with amp():
            tot += model(x, y, skip=skip).item()
    model.train(was)
    return tot / len(batches)


@torch.no_grad()
def knockout(model, batches, amp):
    base = evaluate(model, batches, amp)
    return base, {(li, ci): evaluate(model, batches, amp, skip={(li, ci)}) - base
                  for li in range(model.L) for ci in range(model.K)}


def nullctx():
    return torch.autocast("cpu", enabled=False)

# --------------------------------------------------------------------------- data (imported lazily: chat/parts tools do not need it)


def _v3():
    import seed_v3_gpu as v3
    return v3


def make_fastmix_cls():
    v3 = _v3()

    class FastMix(v3.TextMix):
        """v3's TextMix (same sources/shares); training rows tokenised with one encode_batch call per batch.
        Validation rows still come from v3's own builder, so val_mix is identical to v3 and older v4 runs."""

        doc_align = False   # run4: start every read right after a <|endoftext|> (whole transcripts, not mid-document)

        def _text(self, s, rng, held):
            if not self.doc_align:
                return super()._text(s, rng, held)
            lo, hi = (s["cut"], s["size"]) if held else (0, s["cut"])
            off = int(rng.integers(lo, max(lo + 1, hi - self.rb)))
            s["f"].seek(off); raw = s["f"].read(self.rb)
            k = raw.find(b"<|endoftext|>")
            if 0 <= k < len(raw) - 64:
                raw = raw[k + 13:]
            else:
                sp = raw.find(b" ")
                raw = raw[sp + 1:] if 0 <= sp < 64 else raw
            return raw.decode("utf-8", errors="ignore").replace("\r\n", "\n").replace("\r", "\n")

        def fast_batch(self, B, T, rng, base_tok):
            which = rng.choice(len(self.src), size=B, p=self.p)
            rows = [[] for _ in range(B)]
            need = list(range(B))
            while need:
                pieces, owner = [], []
                for i in need:
                    for j, pc in enumerate(self._text(self.src[which[i]], rng, False).split(self.SEP)):
                        pieces.append(pc); owner.append((i, j > 0))
                for (i, eos_first), e in zip(owner, base_tok.encode_batch(pieces)):
                    if eos_first:
                        rows[i].append(self.eos)
                    rows[i].extend(e.ids)
                need = [i for i in need if len(rows[i]) < T + 1]
            arr = torch.tensor([r[:T + 1] for r in rows], dtype=torch.long)
            return arr[:, :-1].contiguous(), arr[:, 1:].contiguous()
    return FastMix


class Prefetcher:
    def __init__(self, fn, depth=6, pin=True):
        self.fn, self.q, self.pin, self.err, self.step = fn, queue.Queue(maxsize=depth), pin, None, 0
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            try:
                x, y = self.fn(self.step)
                if self.pin:
                    x, y = x.pin_memory(), y.pin_memory()
                self.q.put((x, y))
            except Exception as e:
                self.err = e; self.q.put(None); return

    def get(self, dev):
        item = self.q.get()
        if item is None:
            raise RuntimeError(f"data thread failed: {self.err!r}")
        return item[0].to(dev, non_blocking=True), item[1].to(dev, non_blocking=True)


def pick_precision(dev, log):
    """On GTX 16xx (no tensor cores) fp16 GEMMs are ~8x SLOWER than fp32. Time both once; use the faster."""
    if dev.type != "cuda":
        return False
    def t(dt):
        x = torch.randn(4096, 256, device=dev, dtype=dt); w = torch.randn(1024, 256, device=dev, dtype=dt)
        for _ in range(3):
            F.linear(x, w)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(20):
            F.linear(x, w)
        torch.cuda.synchronize(); return time.perf_counter() - t0
    t16, t32 = t(torch.float16), t(torch.float32)
    use = t16 * 1.3 < t32
    log(f"precision check: fp16 matmul {t16*50:.2f} ms vs fp32 {t32*50:.2f} ms -> {'fp16+GradScaler' if use else 'fp32'}")
    return use


def best_perm(C, fixed=()):
    """Permutation of columns maximising sum C[i, perm[i]] (K <= 7: exhaustive; else greedy). fixed = index stays."""
    K = C.shape[0]
    free = [i for i in range(K) if i not in fixed]
    if K <= 7:
        best, bp = -1e9, list(range(K))
        for pf in itertools.permutations(free):
            perm = list(range(K))
            for i, j in zip(free, pf):
                perm[i] = j
            s = sum(C[i, perm[i]] for i in range(K))
            if s > best + 1e-9:
                best, bp = s, perm
        return bp
    perm, used = list(range(K)), set(fixed)
    for i in free:
        j = max((j for j in free if j not in used), key=lambda j: C[i, j]); perm[i] = j; used.add(j)
    return perm


# --------------------------------------------------------------------------- loading anything (used by chat.py / parts_tool.py)


DECODE_PRIOR_PATHS = [Path(r"D:\seed_v4\decode_prior\seed-bpe4096-v1_cluster_prior.pt"),
                      HERE / "tokenizers" / "seed-bpe4096-v1" / "cluster_prior.pt"]


def load_decode_prior(model=None, path=None):
    """Decode-side fix for the brace bug. The seed-socket-v1 anchor came from a TinyStories-only run, so ~183 tokens never
    seen there ('{', '}', '>', '_', '=', '[', most raw bytes) have near-identical frozen (un)embedding rows (cos ~0.999) and a
    socket seed cannot tell them apart: it gets the group's total probability right but splits it evenly, so '{' comes out
    as a random byte (shown as U+FFFD). The prior (built by build_decode_prior.py from real text) re-splits the group's mass by
    bigram frequency given the previous token. It only moves probability inside that group."""
    for p in ([Path(path)] if path else DECODE_PRIOR_PATHS):
        if p.exists():
            d = torch.load(p, map_location="cpu", weights_only=False)
            sha = ((getattr(model, "tokenizer", None) or {}).get("sha256")) if model is not None else None
            if sha and d.get("tokenizer_sha256") and sha != d["tokenizer_sha256"]:
                continue
            return d
    return None


def prior_bias(pr, prev_id, V, device):
    b = torch.zeros(V, device=device)
    if "bigram" in pr:
        b[pr["group_ids"].to(device)] = pr["bigram"][prev_id].to(device).float()
    else:
        b[:pr["bias"].numel()] += pr["bias"].to(device)
    return b


def load_model_any(path, use_ema=True, device="cpu"):
    """latest.pt / snapshot .pt (EMA weights by default) or a tree part (.json). Returns (model, info)."""
    path = Path(path)
    if path.suffix == ".json":
        m, man = load_tree(path)
        m.decode_prior = load_decode_prior(m)
        return m.to(device).eval(), dict(kind="tree", src=str(path), step=man.get("step"), ema=man.get("ema"))
    c = torch.load(path, map_location="cpu", weights_only=False)
    m = SeedV4.from_structure(c["structure"])
    sd = {k: v.float() if v.is_floating_point() else v for k, v in c["model"].items()}
    m.load_state_dict(sd)
    used = False
    if use_ema and c.get("ema"):
        with torch.no_grad():
            for n, p in m.named_parameters():
                if n in c["ema"] and c["ema"][n].shape == p.shape:
                    p.copy_(c["ema"][n].float()); used = True
    step = c.get("S", {}).get("step", c.get("step"))
    m.decode_prior = load_decode_prior(m)
    return m.to(device).eval(), dict(kind="checkpoint", src=str(path), step=step, ema=used)


def install_parts_full(model, part_files, log=print):
    """Install organ parts at full strength (alpha 1) for inference/scoring; cell parts swap into their slot."""
    done = []
    for pf in part_files:
        man, t = read_part(pf)
        errs, warns = validate_part(model, man, t)
        if errs:
            raise SystemExit(f"{pf}: cannot install: {'; '.join(errs)}")
        for w in warns:
            log(f"note: {pf}: {w}")
        dev = model.emb.weight.device
        t = {k: v.to(dev) for k, v in t.items()}
        if man["kind"] == "organ":
            oid = install_organ(model, man, t)
            k = model.organ_ids.index(oid)
            for l in model.layers:
                l.cells[k].alpha.fill_(1.0)
            done.append(f"organ {oid}")
        else:
            loc = model.find_cell(man.get("target") or man["id"])
            li, ci = loc if loc else (min(int(man.get("layer", 0)), model.L - 1), 0)
            load_cell_weights(model, li, ci, t)
            done.append(f"cell {man['id']} -> L{li}.{model.organ_ids[ci]}")
    return done


def find_tokenizer(model, override=None):
    from tokenizers import Tokenizer
    cands = [override] if override else []
    tdir = model.tokenizer.get("dir")
    if tdir:
        cands.append(Path(tdir) / "tokenizer.json")
        cands.append(HERE / tdir / "tokenizer.json")
    cands += [HERE / "data" / "tokenizer_4096.json", HERE / "tokenizers" / "seed-bpe4096-v1" / "tokenizer.json"]
    for cnd in cands:
        if cnd and Path(cnd).exists():
            tk = Tokenizer.from_file(str(cnd))
            sha = model.tokenizer.get("sha256")
            if sha and file_sha(Path(cnd)) != sha:
                print(f"warning: {cnd} is not the tokenizer this model was trained with")
            return tk
    raise SystemExit("tokenizer not found; pass --tokenizer")


# --------------------------------------------------------------------------- seed-socket-v1: portable organs
ORGAN_FORMAT = "seed-organ/1.1"          # 1.1 = 1 + optional rehearsal cache (cache.* tensors, manifest "rehearsal")
ORGAN_FORMATS = ("seed-organ/1", "seed-organ/1.1")   # both import
AUTO_ALIGN_MARGIN = 0.1   # Procrustes must beat the identity map's fit by this much to be used (per layer, per side)
DEFAULT_SOCKET_DIR = str(HERE / "socket" / "seed-socket-v1")


def socket_files(sd):
    sd = Path(sd)
    return sd / "socket.json", sd / "anchor_emb.safetensors", sd / "calib_tokens.safetensors"


def create_socket(sd, emb, valid_tokens, tok_man, n_seq=16, T=128, seed=0, source="", readout_gain=None):
    """Publish the socket: spec + frozen anchor (un)embedding + calibration token set (all versioned, sha-pinned)."""
    from safetensors.torch import save_file
    spec_p, anc_p, cal_p = socket_files(sd)
    if spec_p.exists():
        return json.loads(spec_p.read_text(encoding="utf-8"))
    Path(sd).mkdir(parents=True, exist_ok=True)
    emb = emb.detach().float().cpu().contiguous()
    g = (torch.ones(emb.shape[1]) if readout_gain is None else readout_gain.detach().float().cpu()).contiguous()
    save_file({"emb": emb, "readout_gain": g}, str(anc_p), metadata={"socket": SOCKET_VERSION})
    r = np.random.default_rng(seed)
    st = r.integers(0, len(valid_tokens) - T - 1, size=n_seq)
    cal = torch.tensor(np.stack([np.asarray(valid_tokens[i:i + T], dtype=np.int64) for i in st]))
    save_file({"tokens": cal.to(torch.int32)}, str(cal_p), metadata={"socket": SOCKET_VERSION})
    spec = dict(socket=SOCKET_VERSION, width=int(emb.shape[1]), vocab=int(emb.shape[0]),
                tokenizer=dict(id=tok_man["id"], sha256=tok_man["sha256"]),
                anchor=dict(file=anc_p.name, sha256=file_sha(anc_p), source=source,
                            rule="token embedding = anchor.emb (frozen); logits = (rmsnorm(x) * anchor.readout_gain) @ anchor.emb^T (tied, both frozen)"),
                calib=dict(file=cal_p.name, sha256=file_sha(cal_p), n_seq=n_seq, T=T),
                norm="RMSNorm (default eps) with gain fixed at 1 at every cell input (attention and MLP); the readout uses the anchor's frozen readout_gain",
                rope=dict(style="interleaved complex", base=ROPE_BASE, head_dim=SOCKET_HEAD_DIM),
                cell=dict(kv_heads=1, adapters="u = h + bin(ain h) on input; o' = o + bout(aout o) on output; low rank",
                          free="n_head, mlp, rank are free (host pads smaller organs exactly); head_dim fixed"),
                created=time.strftime("%Y-%m-%d %H:%M:%S"))
    spec_p.write_text(json.dumps(spec, indent=1), encoding="utf-8")
    return spec


def load_socket(sd):
    from safetensors.torch import load_file
    spec_p, anc_p, cal_p = socket_files(sd)
    spec = json.loads(spec_p.read_text(encoding="utf-8"))
    if file_sha(anc_p) != spec["anchor"]["sha256"] or file_sha(cal_p) != spec["calib"]["sha256"]:
        raise SystemExit(f"socket files in {sd} do not match socket.json (edited?)")
    an = load_file(str(anc_p))
    spec["_readout_gain"] = an.get("readout_gain", torch.ones(an["emb"].shape[1]))
    return spec, an["emb"], load_file(str(cal_p))["tokens"].long()


@torch.no_grad()
def residual_acts(model, tokens, bs=8):
    """rmsnorm(residual) at every layer input + the final residual, on the socket calibration tokens: (L+1) x (N, d)."""
    was = model.training; model.eval()
    dev = model.emb.weight.device
    tokens = tokens[:, :model.ctx]          # hosts with a shorter context use a prefix of every calibration row
    out = [[] for _ in range(model.L + 1)]
    for i in range(0, len(tokens), bs):
        idx = tokens[i:i + bs].to(dev)
        x = model.emb(idx)
        for li, layer in enumerate(model.layers):
            out[li].append(F.rms_norm(x.float(), (model.d,)).reshape(-1, model.d).cpu())
            x, _, _ = layer(x, model.fc)
        out[-1].append(F.rms_norm(x.float(), (model.d,)).reshape(-1, model.d).cpu())
    model.train(was)
    return [torch.cat(o) for o in out]


def procrustes(Xh, Xd):
    """Orthogonal R minimising ||Xh R - Xd|| (host space -> donor space). Returns R and the fit (1 = perfect)."""
    Xh, Xd = Xh.double(), Xd.double()
    U, _, Vt = torch.linalg.svd(Xh.T @ Xd)
    R = U @ Vt
    fit = 1 - float(((Xh @ R - Xd) ** 2).sum() / (Xd ** 2).sum())
    fit0 = 1 - float(((Xh - Xd) ** 2).sum() / (Xd ** 2).sum())
    return R.float(), fit, fit0



def organ_state(model, k):
    t = {}
    for li, layer in enumerate(model.layers):
        for kk, v in cell_state(layer.cells[k]).items():
            t[f"L{li}.{kk}"] = v
        t[f"L{li}.router"] = layer.router[k]
    return t


def bundle_sha_payload(man):
    m = {k: v for k, v in man.items() if k != "signature"}
    return json.dumps(m, sort_keys=True, separators=(",", ":")).encode()


def harvest_bundle(model, k, stem, calib_tokens, meta=None, log=print, signer=None, cache=None, rehearsal=None):
    """Export organ column k as a portable bundle: weights + donor calibration activations + rehearsal cache + manifest."""
    from safetensors.torch import save_file
    if not model.socket:
        raise ValueError("only seed-socket models make portable organs")
    stem = Path(stem)
    if stem.suffix in (".json", ".safetensors"):
        stem = stem.with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    st_p, js_p = Path(str(stem) + ".safetensors"), Path(str(stem) + ".json")   # names like name-1.0.0 keep their dots
    was = [float(l.cells[k].alpha) for l in model.layers]
    t = {n: v.detach().float().cpu().contiguous() for n, v in organ_state(model, k).items()}
    with torch.no_grad():      # calibration = the trunk's own residual stream, WITHOUT this organ (what a host looks like)
        for l in model.layers:
            l.cells[k].alpha.fill_(0.0)
        acts = residual_acts(model, calib_tokens) if model.K > 1 else None
        for l, a_ in zip(model.layers, was):
            l.cells[k].alpha.fill_(a_)
    if acts is None:   # single-organ seed: there is no trunk without it; use its own stream
        acts = residual_acts(model, calib_tokens)
    for j, a in enumerate(acts):
        t[f"calib.L{j}"] = a.half().contiguous()
    calib_T = min(calib_tokens.shape[1], model.ctx)
    if cache is not None and len(cache):
        t.update(cache.to_tensors())
    ok = safe_save(t, st_p, log, margin_mb=50, writer=lambda obj, p: save_file(obj, str(p), metadata={"format": ORGAN_FORMAT}))
    if not ok:
        return None
    n_par = int(sum(v.numel() for n, v in t.items() if not n.startswith(("calib.", "cache."))))
    meta = dict(meta or {})
    meta.pop("price", None)
    oid = model.organ_ids[k]
    om = model.organ_meta.get(oid, {})
    man = dict(format=ORGAN_FORMAT, socket=model.socket,
               name=meta.pop("name", om.get("name", f"organ-{oid}-{model.lineage}")), version=meta.pop("version", "0.1.0"),
               description=meta.pop("description", om.get("description", "")), author=meta.pop("author", "unknown"),
               license=meta.pop("license", "CC-BY-4.0"),
               tokenizer=dict(id=model.tokenizer.get("id"), sha256=model.tokenizer.get("sha256")),
               anchor_sha256=model.anchor.get("sha256"), width=model.d,
               cell_profile=dict(n_head=model.ckw["n_head"], head_dim=model.ckw["head_dim"], mlp=model.ckw["mlp"],
                                 rank=model.ckw.get("rank", 8), kv_heads=1),
               n_cells=model.L, depth_hints=[round((li + 0.5) / model.L, 4) for li in range(model.L)],
               alpha_at_export=was, n_params=n_par, gate="plugin-sigmoid" if model.layers[0].cells[k].plugin else "softmax", calib=dict(n_seq=int(calib_tokens.shape[0]), T=int(calib_T)),
               cost=dict(ram_mb=round(n_par * 4 / 2**20, 2), train_vram_mb_est=round(n_par * 16 / 2**20 + 64, 1)),
               donor=dict(shape=model.shape_str(), lineage=model.lineage, organ_id=oid), protected=oid in model.protected_organs,
               file=st_p.name, sha256=file_sha(st_p, n=64), created=time.strftime("%Y-%m-%d %H:%M:%S"),
               rehearsal=rehearsal or cache_manifest(cache, model.tokenizer, included=cache is not None and len(cache) > 0),
               signature=None, **meta)
    if signer:
        man["signature"] = signer(bundle_sha_payload(man))
    tmp = js_p.with_suffix(".json.tmp"); tmp.write_text(json.dumps(man, indent=1), encoding="utf-8"); os.replace(tmp, js_p)
    return js_p


def read_bundle(js_path):
    from safetensors.torch import load_file
    js_path = Path(js_path)
    man = json.loads(js_path.read_text(encoding="utf-8"))
    if man.get("format") not in ORGAN_FORMATS:
        raise ValueError(f"not an organ bundle (format {man.get('format')!r})")
    st_p = js_path.with_name(man["file"])
    if file_sha(st_p, n=64) != man.get("sha256"):
        raise ValueError("weights do not match the manifest sha256 (corrupt or tampered)")
    return man, load_file(str(st_p))


def bundle_compat(model, man):
    errs, warns = [], []
    if not model.socket:
        errs.append("host is not a seed-socket model")
    if man.get("socket") != model.socket:
        errs.append(f"socket {man.get('socket')!r} != host {model.socket!r}")
    ht, bt = model.tokenizer.get("sha256"), (man.get("tokenizer") or {}).get("sha256")
    if not bt or bt != ht:
        errs.append(f"tokenizer {(man.get('tokenizer') or {}).get('id')} != host {model.tokenizer.get('id')}")
    if man.get("anchor_sha256") != model.anchor.get("sha256"):
        errs.append("anchor embedding differs from the host's (different socket anchor)")
    if man.get("width") != model.d:
        errs.append(f"width {man.get('width')} != host {model.d}")
    cp = man.get("cell_profile", {})
    if cp.get("head_dim") != model.ckw["head_dim"]:
        errs.append(f"head_dim {cp.get('head_dim')} != host {model.ckw['head_dim']}")
    for key, hv in (("n_head", model.ckw["n_head"]), ("mlp", model.ckw["mlp"]), ("rank", model.ckw.get("rank", 8))):
        if cp.get(key, 0) > hv:
            errs.append(f"organ {key} {cp.get(key)} > host {hv} (host can only pad smaller cells)")
    if man.get("n_cells") != model.L:
        warns.append(f"organ has {man.get('n_cells')} cells, host has {model.L} layers: cells mapped by relative depth")
    if (man.get("donor") or {}).get("lineage") != model.lineage:
        warns.append("foreign organ (other seed): alignment chosen per --align (identity, Procrustes or auto) on the socket calibration set")
    return errs, warns


def map_layers(hints, L):
    return [min(range(len(hints)), key=lambda j: abs(hints[j] - (li + 0.5) / L)) for li in range(L)]


def pad_cell(t, cp, ckw):
    nh0, nh1, hd = cp["n_head"], ckw["n_head"], ckw["head_dim"]
    m0, m1, r0, r1 = cp["mlp"], ckw["mlp"], cp.get("rank", 8), ckw.get("rank", 8)
    t = dict(t)
    if nh1 > nh0:
        t["q"] = torch.cat([t["q"], t["q"].new_zeros((nh1 - nh0) * hd, t["q"].shape[1])])
        t["proj"] = torch.cat([t["proj"], t["proj"].new_zeros(t["proj"].shape[0], (nh1 - nh0) * hd)], 1)
    if m1 > m0:
        g, u = t["w12"][:m0], t["w12"][m0:]
        z = t["w12"].new_zeros(m1 - m0, t["w12"].shape[1])
        t["w12"] = torch.cat([g, z, u, z])
        t["down"] = torch.cat([t["down"], t["down"].new_zeros(t["down"].shape[0], m1 - m0)], 1)
    if r1 > r0:
        for k in ("ain", "aout"):
            t[k] = torch.cat([t[k], t[k].new_zeros(r1 - r0, t[k].shape[1])])
        for k in ("bin", "bout"):
            t[k] = torch.cat([t[k], t[k].new_zeros(t[k].shape[0], r1 - r0)], 1)
    return t


def rotate_cell(t, Rin, Rout):
    """Map a donor cell into host coordinates. R: host->donor (Xh R ~ Xd); host column = R @ donor column."""
    t = dict(t)
    for k in ("q", "kv", "w12", "ain"):
        t[k] = t[k] @ Rin.T
    t["router"] = t["router"] @ Rin.T
    t["bin"] = Rin @ t["bin"]
    for k in ("proj", "down", "bout"):
        t[k] = Rout @ t[k]
    t["aout"] = t["aout"] @ Rout.T
    return t


@torch.no_grad()
def install_bundle(model, man, t, calib_tokens=None, align=True, host_acts=None, protected=None):
    """Adopt an organ bundle as a new column at alpha 0 (exactly function preserving). Returns (oid, report)."""
    errs, warns = bundle_compat(model, man)
    if errs:
        raise ValueError("; ".join(errs))
    L = model.L
    mp = map_layers(man["depth_hints"], L)
    fits = []
    if align and host_acts is None:
        host_acts = residual_acts(model, calib_tokens)
    if align:   # compare the same (sequence, position) rows: common prefix length
        Td = man["calib"]["T"]; n_seq = man["calib"]["n_seq"]
        Th = host_acts[0].shape[0] // n_seq
        Tc = min(Td, Th)
        cut = lambda X, T_: X.reshape(n_seq, T_, -1)[:, :Tc].reshape(-1, X.shape[-1])
        host_acts = [cut(X, Th) for X in host_acts]
    dev = model.emb.weight.device
    new_cells = []
    for li, layer in enumerate(model.layers):
        j = mp[li]
        ct = {k.split(".", 1)[1]: v.float() for k, v in t.items() if k.startswith(f"L{j}.")}
        ct = pad_cell(ct, man["cell_profile"], model.ckw)
        if align:
            Rin, f_in, f0 = procrustes(host_acts[li], cut(t[f"calib.L{j}"].float(), Td))
            Rout, f_out, f0o = procrustes(host_acts[li + 1], cut(t[f"calib.L{j + 1}"].float(), Td))
            I_ = torch.eye(model.d)
            use_in = align is True or f_in - f0 > AUTO_ALIGN_MARGIN       # "auto": rotate only where it clearly helps
            use_out = align is True or f_out - f0o > AUTO_ALIGN_MARGIN
            ct = rotate_cell(ct, Rin if use_in else I_, Rout if use_out else I_)
            fits.append(dict(layer=li, donor_cell=j, fit_in=round(f_in, 3), fit_in_identity=round(f0, 3), rotated_in=use_in,
                             fit_out=round(f_out, 3), fit_out_identity=round(f0o, 3), rotated_out=use_out))
        c = Cell(model.d, **model.ckw).to(dev)
        load_cell_state(c, ct)
        c.alpha.zero_(); c.adapt = True; c.plugin = True; c.cid = model.new_cid()
        c.protected = bool(man.get("protected") if protected is None else protected)
        layer.cells.append(c); new_cells.append(c)
        layer.router = nn.Parameter(torch.cat([layer.router.data, ct["router"].to(dev).view(1, -1)], 0))
    oid = model.new_oid()
    model.organ_ids.append(oid)
    if new_cells[0].protected:
        model.protected_organs.add(oid)
    model.organ_meta[oid] = dict(name=man.get("name"), version=man.get("version"), sha256=man.get("sha256"),
                                 description=man.get("description"), aligned=str(align), source_lineage=(man.get("donor") or {}).get("lineage"))
    return oid, dict(warnings=warns, layer_map=mp, fits=fits)

# --------------------------------------------------------------------------- rehearsal cache (seed-organ/1.1)
CACHE_SRC = {"task": 0, "core": 1, "synthetic": 2, "bundle": 3, "harvest": 4}
CACHE_SRC_INV = {v: k for k, v in CACHE_SRC.items()}
CACHE_SELECTION = ("per-sequence organ gap: score = (loss without organ - loss with organ) + 0.5 * loss with organ, "
                   "masked to the organ's target tokens; only sequences where the organ helps (gap > 0); garbage filter = "
                   "loss-with-organ above the running 98th percentile is dropped; near-duplicates (MinHash of token 3-grams, "
                   ">= 0.8 similar) keep only the higher score; coverage = 8 difficulty strata (running score quantiles) with "
                   "equal quotas; inside a stratum a weighted reservoir (A-Res key u^(1/score)) decides who stays")
_MH_P = None


def _minhash(tokens, n_hash=16):
    """tokens (B, T) int64 numpy -> (B, n_hash) int64 MinHash signatures of token 3-gram sets."""
    global _MH_P
    if _MH_P is None or _MH_P[0].shape[0] != n_hash:
        r = np.random.default_rng(1234567)
        _MH_P = (r.integers(1, 2**62, size=n_hash, dtype=np.int64).astype(np.uint64) | np.uint64(1),
                 r.integers(0, 2**62, size=n_hash, dtype=np.int64).astype(np.uint64))
    a_, b_ = _MH_P
    t = tokens.astype(np.uint64)
    g = (t[:, :-2] << np.uint64(26)) ^ (t[:, 1:-1] << np.uint64(13)) ^ t[:, 2:]          # (B, T-2) 3-gram ids
    with np.errstate(over="ignore"):
        h = (g[:, :, None] * a_[None, None, :] + b_[None, None, :]) >> np.uint64(1)       # wrapping multiply-add hash
    return h.min(1).astype(np.int64)


class RehearsalCache:
    """Bounded per-organ cache of the hardest sequences for that organ (token ids int16 + packed loss mask).

    Online: add() takes scored candidates (loss with / without the organ); the cache keeps a diverse, stratified,
    priority-reservoir sample of the sequences where the organ matters most. sample() gives rehearsal batches."""

    def __init__(self, cap=1024, T=128, n_strata=8, dedup=0.8, garbage_pct=98.0, n_hash=16, seed=0, pinned=False,
                 provenance="", data_license=""):
        self.cap, self.T, self.n_strata, self.dedup, self.garbage_pct, self.n_hash = int(cap), int(T), n_strata, dedup, garbage_pct, n_hash
        self.tokens = torch.zeros(0, self.T + 1, dtype=torch.int16)
        self.mask = torch.zeros(0, self.T, dtype=torch.bool)
        self.score = torch.zeros(0); self.key = torch.zeros(0); self.source = torch.zeros(0, dtype=torch.uint8)
        self.fp = torch.zeros(0, n_hash, dtype=torch.int64)
        self.shist, self.lhist = [], []
        self.seen, self.rej = 0, dict(dup=0, garbage=0, unhelpful=0, low=0)
        self.baseline = None            # organ's loss on its held-out cache rows when formed / installed (refresher reference)
        self.pinned = pinned            # True = content fixed (imported / task cache); online core updates skip it
        self.provenance, self.data_license = provenance, data_license
        self.g = np.random.default_rng(seed)

    def __len__(self):
        return int(self.tokens.shape[0])

    @staticmethod
    def seq_bytes(T):
        return 2 * (T + 1) + (T + 7) // 8 + 2 + 1      # int16 tokens + packed mask + fp16 score + source

    def nbytes(self):
        return len(self) * self.seq_bytes(self.T)

    def _edges(self):
        if len(self.shist) < 32:
            return None
        return np.quantile(np.asarray(self.shist[-4096:]), np.linspace(0, 1, self.n_strata + 1)[1:-1])

    def _strata(self, sc, edges):
        return np.zeros(len(sc), dtype=np.int64) if edges is None else np.searchsorted(edges, np.asarray(sc))

    def add(self, x_full, mask, l_with, l_without, source="task"):
        """x_full (B, T+1) token ids, mask (B, T) bool (loss positions), l_with / l_without (B,) masked mean losses."""
        x_full = x_full[:, :self.T + 1].detach().cpu().long(); mask = mask[:, :x_full.shape[1] - 1].detach().cpu().bool()
        if x_full.shape[1] < self.T + 1:   # shorter rows: pad with token 3 (<eos>), mask off
            padn = self.T + 1 - x_full.shape[1]
            x_full = torch.cat([x_full, torch.full((x_full.shape[0], padn), 3, dtype=torch.long)], 1)
            mask = torch.cat([mask, torch.zeros(mask.shape[0], padn, dtype=torch.bool)], 1)
        lw, lo = l_with.detach().float().cpu(), l_without.detach().float().cpu()
        fps = torch.from_numpy(_minhash(x_full.numpy(), self.n_hash))
        added = 0
        for b in range(x_full.shape[0]):
            self.seen += 1
            gap, l1 = float(lo[b] - lw[b]), float(lw[b])
            if not (mask[b].any() and math.isfinite(gap) and math.isfinite(l1)):
                continue
            if gap <= 0:
                self.rej["unhelpful"] += 1; continue
            self.lhist.append(l1); self.lhist = self.lhist[-4096:]
            if len(self.lhist) >= 64 and l1 > np.percentile(self.lhist, self.garbage_pct):
                self.rej["garbage"] += 1; continue
            s = gap + 0.5 * l1
            self.shist.append(s); self.shist = self.shist[-4096:]
            key = float(self.g.random() ** (1.0 / max(s, 1e-4)))
            if len(self):
                sim = (self.fp == fps[b]).float().mean(1)
                j = int(sim.argmax())
                if float(sim[j]) >= self.dedup:
                    if s > float(self.score[j]):
                        self._put(j, x_full[b], mask[b], s, key, source, fps[b]); added += 1
                    else:
                        self.rej["dup"] += 1
                    continue
            if len(self) < self.cap:
                self._append(x_full[b], mask[b], s, key, source, fps[b]); added += 1; continue
            edges = self._edges()
            st = self._strata(self.score.numpy(), edges)
            s_new = int(self._strata([s], edges)[0])
            quota = self.cap / self.n_strata
            cnt = np.bincount(st, minlength=self.n_strata)
            if cnt[s_new] < quota:     # under-filled stratum: evict from the most over-filled one
                pool = np.nonzero(st == int(np.argmax(cnt - quota)))[0]
            else:
                pool = np.nonzero(st == s_new)[0]
            v = int(pool[int(np.argmin(self.key.numpy()[pool]))])
            if cnt[s_new] < quota or key > float(self.key[v]):
                self._put(v, x_full[b], mask[b], s, key, source, fps[b]); added += 1
            else:
                self.rej["low"] += 1
        return added

    def _append(self, t, m, s, key, src, fp):
        self.tokens = torch.cat([self.tokens, t.to(torch.int16)[None]]); self.mask = torch.cat([self.mask, m[None]])
        self.score = torch.cat([self.score, torch.tensor([s])]); self.key = torch.cat([self.key, torch.tensor([key])])
        self.source = torch.cat([self.source, torch.tensor([CACHE_SRC.get(src, 0)], dtype=torch.uint8)])
        self.fp = torch.cat([self.fp, fp[None]])

    def _put(self, j, t, m, s, key, src, fp):
        self.tokens[j] = t.to(torch.int16); self.mask[j] = m; self.score[j] = s; self.key[j] = key
        self.source[j] = CACHE_SRC.get(src, 0); self.fp[j] = fp

    def heldout(self):
        """stable 10% check split (by content hash), never used for tuning."""
        return (self.fp[:, 0] % 10 == 0) if len(self) else torch.zeros(0, dtype=torch.bool)

    def batch(self, n, dev, T=None, split="train", rng=None):
        idx = torch.nonzero(self.heldout() if split == "check" else (~self.heldout() if split == "train" else torch.ones(len(self), dtype=torch.bool))).view(-1)
        if len(idx) == 0:
            idx = torch.arange(len(self))
        rng = rng or self.g
        pick = idx[torch.from_numpy(rng.integers(0, len(idx), size=n))] if split != "check_all" else idx
        return self.rows(pick, dev, T)

    def rows(self, pick, dev, T=None):
        T = min(T or self.T, self.T)
        tk = self.tokens[pick].long()[:, :T + 1]; m = self.mask[pick][:, :T]
        y = tk[:, 1:].clone(); y[~m] = -100
        return tk[:, :-1].to(dev), y.to(dev)

    def stats(self):
        src = {CACHE_SRC_INV[int(v)]: int((self.source == v).sum()) for v in self.source.unique()} if len(self) else {}
        return dict(count=len(self), cap=self.cap, T=self.T, bytes=self.nbytes(), seen=self.seen, rejected=dict(self.rej),
                    score_min=round(float(self.score.min()), 4) if len(self) else None,
                    score_median=round(float(self.score.median()), 4) if len(self) else None,
                    score_max=round(float(self.score.max()), 4) if len(self) else None,
                    heldout=int(self.heldout().sum()), sources=src, pinned=self.pinned, baseline=self.baseline)

    # ---- persistence: checkpoint dict and bundle tensors
    def state(self):
        return dict(meta=dict(cap=self.cap, T=self.T, n_strata=self.n_strata, dedup=self.dedup, garbage_pct=self.garbage_pct,
                              n_hash=self.n_hash, seen=self.seen, rej=self.rej, baseline=self.baseline, pinned=self.pinned,
                              provenance=self.provenance, data_license=self.data_license,
                              shist=self.shist[-1024:], lhist=self.lhist[-1024:]),
                    tensors=self.to_tensors(prefix=""))

    @classmethod
    def from_state(cls, st):
        m = st["meta"]
        c = cls(cap=m["cap"], T=m["T"], n_strata=m["n_strata"], dedup=m["dedup"], garbage_pct=m["garbage_pct"], n_hash=m["n_hash"],
                pinned=m.get("pinned", False), provenance=m.get("provenance", ""), data_license=m.get("data_license", ""))
        c.load_tensors(st["tensors"], prefix="")
        c.seen, c.rej, c.baseline = m["seen"], dict(m["rej"]), m.get("baseline")
        c.shist, c.lhist = list(m.get("shist", [])), list(m.get("lhist", []))
        return c

    def to_tensors(self, prefix="cache."):
        bits = np.packbits(self.mask.numpy().astype(np.uint8), axis=1) if len(self) else np.zeros((0, (self.T + 7) // 8), np.uint8)
        return {prefix + "tokens": self.tokens.contiguous(), prefix + "maskbits": torch.from_numpy(np.ascontiguousarray(bits)),
                prefix + "score": self.score.half().contiguous(), prefix + "source": self.source.contiguous()}

    def load_tensors(self, t, prefix="cache."):
        self.tokens = t[prefix + "tokens"].to(torch.int16)
        self.T = int(self.tokens.shape[1] - 1)
        bits = t[prefix + "maskbits"].numpy()
        self.mask = torch.from_numpy(np.unpackbits(bits, axis=1)[:, :self.T].astype(bool)) if len(bits) else torch.zeros(0, self.T, dtype=torch.bool)
        self.score = t[prefix + "score"].float(); self.source = t[prefix + "source"].to(torch.uint8)
        self.key = torch.from_numpy(self.g.random(len(self.score)) ** (1.0 / self.score.clamp_min(1e-4).numpy())).float()
        self.fp = torch.from_numpy(_minhash(self.tokens.long().numpy(), self.n_hash)) if len(self) else torch.zeros(0, self.n_hash, dtype=torch.int64)
        self.cap = max(self.cap, len(self))

    @classmethod
    def from_bundle(cls, man, t):
        if "cache.tokens" not in t:
            return None
        r = man.get("rehearsal") or {}
        c = cls(cap=r.get("cap", r.get("count", 1024)), T=r.get("T", 128), pinned=True,
                provenance=r.get("data_provenance", ""), data_license=r.get("data_license", ""))
        c.load_tensors(t)
        return c


def cache_cap(seqs, mb, T):
    return int(max(16, min(seqs, mb * 2**20 // RehearsalCache.seq_bytes(T))))


@torch.no_grad()
def organ_gap(model, k, x, y, amp=nullctx):
    """per-sequence masked mean loss WITH organ column k and WITHOUT it (skip)."""
    was = model.training; model.eval()
    with amp():
        l1 = model(x, y, per_token=True).float().view(x.shape[0], -1)
        l0 = model(x, y, skip={(li, k) for li in range(model.L)}, per_token=True).float().view(x.shape[0], -1)
    model.train(was)
    m = (y != -100).float(); n = m.sum(1).clamp_min(1)
    return (l1 * m).sum(1) / n, (l0 * m).sum(1) / n


def cache_feed(model, k, cache, x, y, source, amp=nullctx):
    """score a batch for organ k and offer it to the cache. x (B,T), y (B,T) with -100 = not a target."""
    lw, lo = organ_gap(model, k, x, y, amp)
    last = torch.where(y[:, -1:] == -100, torch.full_like(y[:, -1:], 3), y[:, -1:])   # rebuild the T+1 token row
    full = torch.cat([x, last], 1)
    return cache.add(full, y != -100, lw, lo, source)


@torch.no_grad()
def cache_loss(model, cache, dev, split="check", n=128, amp=nullctx, skip_k=None, T=None, bs=32, skip=None):
    """mean target-token loss of the model on (a fixed slice of) the cache; skip_k = without organ column k."""
    if cache is None or not len(cache):
        return None
    idx = torch.nonzero(cache.heldout() if split == "check" else torch.ones(len(cache), dtype=torch.bool)).view(-1)
    if len(idx) < 4:
        idx = torch.arange(len(cache))
    idx = idx[:n]
    was = model.training; model.eval()
    tot, cnt = 0.0, 0
    if skip_k is not None:
        skip = {(li, skip_k) for li in range(model.L)}
    for i in range(0, len(idx), bs):
        x, y = cache.rows(idx[i:i + bs], dev, T or model.ctx)
        with amp():
            l = model(x, y, skip=skip, per_token=True).float()
        m = (y.reshape(-1) != -100)
        tot += float(l[m].sum()); cnt += int(m.sum())
    model.train(was)
    return tot / max(1, cnt)


@torch.no_grad()
def synth_cache(model, k, calib_tokens, dev, n=256, T=128, prefix=8, cap=512, log=print, amp=nullctx):
    """Synthetic rehearsal when a bundle ships no cache: sample from the host WITH the organ (calibration-set
    prefixes), keep the samples where the organ matters most. Weaker than real data (the organ rehearses itself)."""
    cache = RehearsalCache(cap=min(cap, n), T=T, pinned=True, provenance="synthetic: sampled from the organ itself")
    was = model.training; model.eval()
    P = calib_tokens[:, :prefix].long().to(dev)
    made = 0
    while made < n:
        ids = P[torch.randint(0, P.shape[0], (min(32, n - made),))]
        with amp():
            out = model.generate(ids, T + 1 - prefix, temp=1.0, top_k=0, top_p=0.95, rep_pen=None)
        x, y = out[:, :-1], out[:, 1:].clone()
        y[:, :prefix - 1] = -100      # only the generated part is a target
        cache_feed(model, k, cache, x, y, "synthetic", amp)
        made += ids.shape[0]
    model.train(was)
    log(f"synthetic rehearsal cache: {len(cache)} sequences from {made} samples")
    return cache


def cache_manifest(cache, tok_info, included=True, mode="full", note="", reason=None):
    if not included or cache is None:
        return dict(included=False, mode="none", reason=reason or "author chose not to ship training data",
                    fallback="synthetic rehearsal sampled from the organ itself at import")
    st = cache.stats()
    return dict(included=True, mode=mode, count=st["count"], cap=cache.cap, T=cache.T, bytes=st["bytes"],
                tensors=["cache.tokens (int16 [N,T+1])", "cache.maskbits (uint8 packbits [N,ceil(T/8)], 1 = target token)",
                         "cache.score (fp16 [N])", "cache.source (uint8 [N])"],
                sources=st["sources"], source_codes=CACHE_SRC, selection=CACHE_SELECTION, heldout_rule="minhash[0] % 10 == 0 (never tuned on)",
                tokenizer=dict(id=tok_info.get("id"), sha256=tok_info.get("sha256")),
                data_provenance=cache.provenance or "not declared", data_license=cache.data_license or "same as bundle license",
                baseline=cache.baseline, score_range=[st["score_min"], st["score_median"], st["score_max"]], note=note)


def load_caches(path):
    """{organ_id: RehearsalCache} stored in a checkpoint (trainer latest.pt or an adopted host)."""
    try:
        c = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return {}
    raw = c.get("caches") or (c.get("extra") or {}).get("caches") or {}
    return {oid: RehearsalCache.from_state(st) for oid, st in raw.items()}

# --------------------------------------------------------------------------- trainer


def get_args(argv=None):
    ap = argparse.ArgumentParser(description="Seed v4 FINAL trainer")
    A = ap.add_argument
    A("--data", default=str(HERE / "data")); A("--out", default=str(HERE / "runs" / "run1"))
    A("--parts-dir", default=str(HERE / "parts"), help="parts library: inbox/, exports/, removed/, rejected/, processed/")
    A("--tokenizer-dir", default=str(HERE / "tokenizers" / "seed-bpe4096-v1"), help="standard shared tokenizer (created once)")
    A("--fallback-dir", default=None)
    A("--device", default="cuda"); A("--threads", type=int, default=2, help="keep low: other CPU jobs share the PC")
    A("--priority", default="below_normal", choices=["idle", "below_normal", "normal"])
    A("--minutes", type=float, default=20160); A("--max-steps", type=int, default=0)
    A("--wall-minutes", type=float, default=0); A("--resume", action="store_true"); A("--seed", type=int, default=1234)
    A("--init-tree", default=None, help="fresh run starting from a tree part (.json manifest)")
    A("--init-from", default=None, help="fresh run (new optimiser/schedule) starting from any checkpoint .pt, e.g. one with an adopted organ")
    A("--socket", default="v1", choices=["v1", "off"], help="seed-socket-v1: portable cells (own K/V, adapters, frozen shared anchor)")
    A("--socket-dir", default=DEFAULT_SOCKET_DIR)
    A("--adapter-rank", type=int, default=8)
    A("--transplant-tol", type=float, default=0.01, help="hot transplant kept if core val is not worse than control + tol")
    A("--transplant-no-align", action="store_true")
    # rehearsal caches (seed-organ/1.1): every organ keeps its hardest material and rehearses it
    A("--rehearsal", default="on", choices=["on", "off"], help="off = no rehearsal batches and no refresher (control)")
    A("--rehearsal-rate", type=float, default=0.04, help="share of micro-batches per imported/task organ (scheduled)")
    A("--rehearsal-core-rate", type=float, default=0.01, help="total share for organs whose cache is main-run text")
    A("--rehearsal-max", type=float, default=0.25, help="ceiling on the total rehearsal share (incl. refreshers)")
    A("--rehearsal-tol", type=float, default=0.1, help="refresher when an organ's held-out cache loss rises this much (nats) over its baseline")
    A("--rehearsal-check-every", type=int, default=250); A("--refresh-steps", type=int, default=100)
    A("--cache-every", type=int, default=250, help="online cache update (emergent organs) every N steps")
    A("--cache-seqs", type=int, default=512); A("--cache-mb", type=float, default=4.0)
    # shape
    A("--d", type=int, default=384); A("--ctx", type=int, default=512)
    A("--n-head", type=int, default=4); A("--head-dim", type=int, default=32); A("--mlp", type=int, default=192)
    A("--start-layers", type=int, default=2); A("--start-cells", type=int, default=1)
    A("--fixed-shape", default=None); A("--untie", action="store_true"); A("--softcap", type=float, default=30.0)
    # optimisation
    A("--batch", type=int, default=16); A("--accum", type=int, default=4)
    A("--muon-lr", type=float, default=0.01); A("--adam-lr", type=float, default=1.5e-3)
    A("--muon-wd", type=float, default=0.01); A("--wd", type=float, default=0.05)
    A("--warmup", type=int, default=500); A("--final-decay-frac", type=float, default=0.15)
    A("--final-lr-frac", type=float, default=0.05); A("--clip", type=float, default=1.0); A("--ema", type=float, default=0.999)
    A("--amp", default="auto", choices=["auto", "fp16", "off"])
    A("--grad-ckpt", default="auto", choices=["auto", "on", "off"], help="auto = checkpoint only as many layers as the VRAM cap needs")
    # anti-redundancy
    A("--decor", type=float, default=0.05); A("--balance", type=float, default=0.01)
    # growth cycle
    A("--min-cycle", type=int, default=1500); A("--max-cycle", type=int, default=5000)
    A("--plateau-window", type=int, default=600); A("--plateau-rel", type=float, default=0.01)
    A("--cool-steps", type=int, default=300); A("--cool-frac", type=float, default=0.25)
    A("--rewarm-steps", type=int, default=300); A("--rewarm-floor", type=float, default=0.2)
    A("--trial-steps", type=int, default=300); A("--min-gain", type=float, default=0.0); A("--new-min", type=float, default=0.001)
    A("--depth-ratio", type=float, default=4.0); A("--depth-step", type=int, default=2)
    A("--max-layers", type=int, default=24); A("--max-cells", type=int, default=6)
    A("--max-params", type=float, default=25e6); A("--max-rejects", type=int, default=3)
    # hunger / environment
    A("--mem-cap-mb", type=float, default=2048, help="cap on the MODEL WEIGHTS size (fp32 bytes), not on training overhead")
    A("--vram-cap-mb", type=float, default=5000, help="whole-GPU-usage safety cap for this process")
    A("--rss-warn-mb", type=float, default=8192, help="process RSS safety margin: above = tight, above 1.25x = near cap")
    A("--hunger-fed", type=float, default=0.5); A("--hunger-tight", type=float, default=0.15)
    A("--min-sys-free-mb", type=float, default=1024, help="machine-wide free RAM below this = near cap (other programs need it)")
    # organs / organizer / immune / parts
    A("--organizer-every", type=int, default=3000); A("--immune-every", type=int, default=6000)
    A("--immune-rel", type=float, default=0.002, help="freeloader: switching it off costs < this x loss")
    A("--fade-steps", type=int, default=300); A("--inbox-every", type=int, default=50)
    A("--export-on-growth", action="store_true", help="export every new organ to parts/exports when a widen is kept")
    # data
    A("--mix", action="append", default=None); A("--mix-train-only", action="append", default=None)
    A("--mix-holdout", type=float, default=0.01); A("--decide-on", default="mix", choices=["mix", "ts"])
    A("--ts-boost", type=float, default=0.5); A("--ts-boost-steps", type=int, default=6000)
    # logging
    A("--eval-every", type=int, default=250); A("--save-every", type=int, default=1500)
    A("--snapshot-budget-gb", type=float, default=3.0)
    A("--prompts", default="Once upon a time|Lily wanted to|The little dog")
    # run4 (tools + reasoning) additions; defaults keep older runs unchanged
    A("--mix-read-bytes", type=int, default=768, help="bytes per streamed read from a --mix file")
    A("--mix-doc-align", action="store_true", help="start each read at a document boundary (<|endoftext|>)")
    A("--tool-eval", action="store_true", help="at every save: per-source val (tool/chat val) + Wikipedia-tool sample")
    return ap.parse_args(argv)


def main():
    a = get_args()
    global DEV
    DEV = torch.device("cuda" if a.device == "cuda" and torch.cuda.is_available() else "cpu")
    v3 = _v3(); v3.DEV = torch.device("cpu")
    torch.set_num_threads(a.threads); torch.set_num_interop_threads(1)
    if a.priority != "normal":
        v3.lower_priority(a.priority)
    if DEV.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.vram_cap_mb / (torch.cuda.get_device_properties(0).total_memory / 2**20)))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    PD = Path(a.parts_dir)
    for sub in ("inbox", "exports", "removed", "rejected", "processed", "scores"):
        (PD / sub).mkdir(parents=True, exist_ok=True)
    logf = out / "train.log"

    def log(s):
        line = f"[{time.strftime('%H:%M:%S')}] {s}"
        print(line, flush=True)
        try:
            with open(logf, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    def event(kind, **kw):
        rec = dict(kind=kind, step=step, shape=model.shape_str(), params=model.n_params(), **kw)
        log("EVENT " + json.dumps(rec))
        try:
            with open(out / "events.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            pass

    use_amp = (a.amp == "fp16" and DEV.type == "cuda") or (a.amp == "auto" and pick_precision(DEV, log))
    amp = (lambda: torch.autocast("cuda", dtype=torch.float16)) if use_amp else nullctx

    # ---- data + standard tokenizer
    data = Path(a.data)
    meta = json.loads((data / "meta.json").read_text())
    tp = Path(meta["tokenizer"])
    tp = tp if tp.is_absolute() and tp.exists() else (HERE / tp if (HERE / tp).exists() else data / tp.name)
    tman = ensure_std_tokenizer(a.tokenizer_dir, tp)
    base_tok, tman = load_std_tokenizer(a.tokenizer_dir)
    if tman["sha256"] != file_sha(tp):
        raise SystemExit(f"standard tokenizer {tman['id']} differs from the tokenizer the data was built with ({tp})")
    tok_info = dict(id=tman["id"], name=tman["name"], version=tman["version"], sha256=tman["sha256"], dir=str(a.tokenizer_dir))
    tok = v3.GrowTok(base_tok)
    train, valid = v3.Tokens(data / "train.bin"), v3.Tokens(data / "valid.bin")
    todev = lambda bs: [(x.to(DEV), y.to(DEV)) for x, y in bs]
    val_ts = todev(v3.ts_val_batches(valid, a.ctx, tok))
    mix, val_mix = None, None
    if a.mix:
        mix = make_fastmix_cls()(a.mix, tok, tok.token_to_id("<eos>"), a.mix_holdout, read_bytes=a.mix_read_bytes, train_only=a.mix_train_only)
        mix.doc_align = bool(a.mix_doc_align)
        val_mix = todev(mix.val_batches(12, 16, a.ctx))
    dec = val_mix if (mix and a.decide_on == "mix") else val_ts
    use_b = dec[:4]

    # ---- model
    ckw = dict(n_head=a.n_head, head_dim=a.head_dim, mlp=a.mlp)
    sock = None
    if a.socket == "v1":
        ckw.update(socket=True, rank=a.adapter_rank)
        if Path(a.socket_dir, "socket.json").exists():
            sock = load_socket(a.socket_dir)
    fixed = tuple(int(v) for v in a.fixed_shape.split(",")) if a.fixed_shape else None
    ck = out / "latest.pt"
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    S = dict(step=0, cycle_start=0, phase="stable", phase_start=0, rejects=0, prefer_alt=False, hist=[], t_offset=0.0,
             growths=0, fades=[], last_immune=0, last_org=0, last_installed=None, vram_min_mb=0.0,
             reh=dict(credit=0.0, boost={}, refresh={}, n=0, n_micro=0, last_check={}))
    CACHES = {}   # organ id -> RehearsalCache (saved in the checkpoint)
    c = None
    if a.resume and ck.exists():
        c = torch.load(ck, map_location="cpu", weights_only=False)
        model = SeedV4.from_structure(c["structure"])
    elif a.init_tree:
        model, tman_tree = load_tree(a.init_tree)
        if tman_tree.get("tokenizer", {}).get("sha256") not in (None, tok_info["sha256"]):
            raise SystemExit("init tree uses a different tokenizer")
        log(f"starting from tree {a.init_tree} ({model.shape_str()})")
    else:
        L, K = fixed if fixed else (a.start_layers, a.start_cells)
        if a.init_from:
            model, _info = load_model_any(a.init_from, use_ema=True)
            model.train()
            CACHES.update({oid: cc for oid, cc in load_caches(a.init_from).items() if oid in model.organ_ids})
            if CACHES:
                log(f"rehearsal caches from {a.init_from}: " + ", ".join(f"{o}={len(cc)} seqs" for o, cc in CACHES.items()))
            log(f"starting from {a.init_from} ({model.shape_str()}, organs {model.organ_ids})")
        else:
            model = SeedV4(meta["vocab"], a.d, a.ctx, L, K, ckw, tie=True if a.socket == "v1" else not a.untie, softcap=a.softcap)
            if a.socket == "v1":
                if sock is None:
                    raise SystemExit(f"socket files missing in {a.socket_dir} (create them with parts_tool.py make-socket)")
                spec, anchor, _ = sock
                if a.d != spec["width"] or tuple(anchor.shape) != tuple(model.emb.weight.shape) or a.head_dim != spec["rope"]["head_dim"]:
                    raise SystemExit(f"socket {spec['socket']} needs d={spec['width']}, head_dim={spec['rope']['head_dim']}")
                if spec["tokenizer"]["sha256"] != tok_info["sha256"]:
                    raise SystemExit("socket tokenizer differs from the data tokenizer")
                with torch.no_grad():
                    model.emb.weight.copy_(anchor); model.nf.copy_(spec["_readout_gain"])
                model.anchor = dict(id=spec["socket"], sha256=spec["anchor"]["sha256"])
    model.tokenizer = tok_info
    if model.socket and sock is None:
        raise SystemExit(f"this seed is a {model.socket} model but the socket files are missing in {a.socket_dir}")
    model = model.to(DEV)
    opts = build_opts(model, a)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = EMA(model, a.ema)
    if c is not None:
        model.load_state_dict(c["model"])
        for o, sd in zip(opts, c["opts"]):
            o.load_state_dict(sd)
        for o, base in zip(opts, (a.muon_lr, a.adam_lr)):
            for g in o.param_groups:
                g["base_lr"] = base
        if c.get("scaler"):
            scaler.load_state_dict(c["scaler"])
        ema.load_by_name(model, c["ema"]); ema.n = c.get("ema_n", 0)
        S.update(c["S"])
        S.setdefault("reh", dict(credit=0.0, boost={}, refresh={}, n=0, n_micro=0, last_check={}))
        CACHES.update({oid: RehearsalCache.from_state(st_) for oid, st_ in (c.get("caches") or {}).items()})
        if S["phase"] in ("trial", "cool"):
            S["phase"], S["phase_start"] = "rewarm", S["step"]
        try:
            rng.bit_generator.state = c["np_rng"]; torch.set_rng_state(c["torch_rng"])
        except Exception:
            pass
        log(f"resumed {ck} at step {S['step']} shape {model.shape_str()} organs {model.organ_ids}")
        c = None
    step = S["step"]
    t0 = time.time(); t_off = S["t_offset"]; sess0 = time.time()

    # ---- memory: selective gradient checkpointing (only as many layers as the VRAM cap needs)
    def probe_memory(tag=""):
        L = model.L
        if DEV.type != "cuda" or a.grad_ckpt != "auto":
            model.ckpt_layers = L if a.grad_ckpt == "on" else 0
            return
        x = torch.randint(0, model.vocab, (a.batch, a.ctx), device=DEV)

        def run(nck):
            model.ckpt_layers = nck
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            try:
                with torch.enable_grad():   # may be called from inside no_grad code (fades)
                    with amp():
                        l = model(x, x, stats=dict(decor=[], bal=[]))
                    l.backward(); l = None
                pk = torch.cuda.max_memory_allocated()
            except torch.cuda.OutOfMemoryError:
                pk = float("inf")
            model.zero_grad(set_to_none=True); torch.cuda.empty_cache()
            return pk
        p_all = run(L)
        p_m1 = run(L - 1) if L > 1 else p_all
        delta = max(p_m1 - p_all, 2**20) if math.isfinite(p_m1) else float("inf")
        n_state = sum(p.numel() for p in model.parameters()) * 4 * 3  # optimiser/EMA buffers not yet touched
        missing = n_state if step == 0 else 0
        budget = a.vram_cap_mb * 0.85 * 2**20 - p_all - missing
        n_free = 0 if not math.isfinite(delta) or budget <= 0 else int(min(L, budget // delta))
        model.ckpt_layers = L - n_free
        S["vram_min_mb"] = (p_all + missing) / 2**20
        torch.cuda.reset_peak_memory_stats()
        log(f"memory probe{tag} {model.shape_str()} batch {a.batch}x{a.ctx}: all-ckpt peak {p_all/2**20:.0f} MB, "
            f"+{delta/2**20:.0f} MB per uncheckpointed layer -> checkpointing {model.ckpt_layers}/{L} layers")

    def rebuild(reset=()):
        nonlocal opts
        opts = build_opts(model, a, old=opts, reset=reset)
        ema.sync(model, reset)
        probe_memory()

    probe_memory(" (start)")
    rss0 = rss_mb()

    def hunger():
        """RAM cap = model weights; RSS only a safety margin; VRAM cap = whole-process GPU use."""
        w_mb = model.n_params() * 4 / 2**20
        r_head = 1 - w_mb / a.mem_cap_mb
        r = rss_mb()
        if r == r and r > a.rss_warn_mb:
            r_head = min(r_head, a.hunger_tight - 0.01 if r > 1.25 * a.rss_warn_mb else (a.hunger_fed + a.hunger_tight) / 2)
            if not S.get("rss_warned"):
                S["rss_warned"] = True; log(f"WARNING: process RSS {r:.0f} MB > {a.rss_warn_mb:.0f} MB safety margin")
        if sys_avail_mb() < a.min_sys_free_mb:
            r_head = min(r_head, 0.1)
        v_head = 1 - S["vram_min_mb"] / a.vram_cap_mb if DEV.type == "cuda" else 1.0
        f = min(r_head, v_head)
        return ("well_fed" if f > a.hunger_fed else "tight" if f > a.hunger_tight else "near_cap"), f

    # ---- save
    def save(tag=None):
        S["step"] = step; S["t_offset"] = t_off + (time.time() - t0)
        obj = dict(structure=model.structure(), model=model.state_dict(), opts=[o.state_dict() for o in opts],
                   scaler=scaler.state_dict(), ema=ema.state_by_name(model), ema_n=ema.n, S=S, args=vars(a),
                   np_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                   caches={oid: cc.state() for oid, cc in CACHES.items() if oid in model.organ_ids})
        p = safe_save(obj, ck, log, a.fallback_dir)
        if p and tag:
            sd = out / "snapshots"
            used = sum(f.stat().st_size for f in sd.glob("*.pt")) / 2**30 if sd.exists() else 0.0
            if used < a.snapshot_budget_gb:
                snap = dict(structure=obj["structure"], step=step,
                            model={k: (v.half() if v.is_floating_point() else v) for k, v in obj["model"].items()},
                            ema={k: v.half() for k, v in obj["ema"].items()})
                safe_save(snap, sd / f"{tag}.pt", log, None, retries=2)
            else:
                log(f"snapshot budget reached ({used:.2f} GB): not writing {tag} (nothing is deleted)")
        return p

    # ---- schedule
    def lr_mult():
        st = step
        m = min(1.0, (st + 1) / a.warmup)
        ph, ps = S["phase"], S["phase_start"]
        if ph == "rewarm":
            m *= a.rewarm_floor + (1 - a.rewarm_floor) * min(1.0, (st - ps) / max(1, a.rewarm_steps))
        elif ph == "cool":
            m *= 1 - (1 - a.cool_frac) * min(1.0, (st - ps) / max(1, a.cool_steps))
        elif ph == "trial":
            t = (st - ps) / max(1, a.trial_steps)
            if t < 1 / 3:
                m *= a.rewarm_floor + (1 - a.rewarm_floor) * 3 * t
            elif t > 2 / 3:
                m *= 1 - (1 - a.cool_frac) * min(1.0, 3 * (t - 2 / 3))
        prog = step / a.max_steps if a.max_steps else (t_off + time.time() - t0) / (a.minutes * 60)
        if prog > 1 - a.final_decay_frac:
            m *= max(a.final_lr_frac, 1 - (prog - (1 - a.final_decay_frac)) / a.final_decay_frac * (1 - a.final_lr_frac))
        return m, prog

    # ---- batches on a background thread
    def make_batch(st):
        B = a.batch
        if not mix:
            return train.batch(B, a.ctx, rng)
        p_ts = a.ts_boost * max(0.0, 1 - st / max(1, a.ts_boost_steps))
        k = int(rng.binomial(B, p_ts)) if p_ts > 0 else 0
        parts = []
        if k:
            parts.append(train.batch(k, a.ctx, rng))
        if B - k:
            parts.append(mix.fast_batch(B - k, a.ctx, rng, base_tok))
        return torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])

    # ---- run4: tool/chat eval (per-source held-out loss + a Wikipedia-tool sample, same prompt as the 100M)
    TOOL_SYS_100M = ('<|system|>You are a helpful assistant. You can call these tools:\n'
                     '{"name": "wiki_search", "description": "Search Wikipedia article introductions", "parameters": {"query": "string"}}\n'
                     '{"name": "calculator", "description": "Evaluate a math expression", "parameters": {"expression": "string"}}<|end|>\n')
    TOOL_SYS_DATA = ('<|system|>You are a helpful assistant. You can call these tools:\n'
                     '{"name": "calculator", "description": "Evaluate an arithmetic expression exactly", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "e.g. 12*(3+4)"}}, "required": ["expression"]}}\n'
                     '{"name": "wiki_search", "description": "Search Wikipedia article introductions and return the best matches", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "what to look up"}}, "required": ["query"]}}<|end|>\n')
    TOOL_Q = "Who was Albert Einstein?"
    src_val = {}
    if a.tool_eval and mix:
        r_ = np.random.default_rng(991)
        for k_, s_ in enumerate(mix.src):
            if s_["val"]:
                src_val[s_["name"]] = todev([mix.batch(8, a.ctx, r_, held=True, which=[k_] * 8) for _ in range(3)])
    tool_csv = out / "tool_metrics.csv"

    def tool_eval():
        """Called at save time with the EMA weights swapped in."""
        import re as _re
        if model.__dict__.get("decode_prior") is None:
            model.decode_prior = load_decode_prior(model)
        res = {n_: round(evaluate(model, b_, amp), 4) for n_, b_ in src_val.items()}
        chat = [v_ for n_, v_ in res.items() if "TinyStories" not in n_ and "ts_" not in n_]
        chat_mean = round(float(np.mean(chat)), 4) if chat else float("nan")
        outs = {}
        for tag, sysp in (("data", TOOL_SYS_DATA), ("100m", TOOL_SYS_100M)):
            ids = base_tok.encode(sysp + "<|user|>" + TOOL_Q + "<|end|>\n<|assistant|>").ids
            ids = ids[-(a.ctx - 100):]
            with amp():
                o_ = model.generate(torch.tensor([ids], device=DEV), 90, temp=0.5)[0].tolist()
            outs[tag] = base_tok.decode(o_[len(ids):]).split("<|end|>")[0]
        qs = {t_: (_re.search(r'"query":\s*"([^"]*)"', o_) or [None, None])[1] for t_, o_ in outs.items()}
        ok = {t_: bool(q_ and "einstein" in q_.lower()) for t_, q_ in qs.items()}
        log(f"TOOLVAL step {step} | tool/chat val {chat_mean} | " + " ".join(f"{n_} {v_}" for n_, v_ in res.items()))
        for t_, o_ in outs.items():
            log(f"TOOLSAMPLE[{t_}] query={qs[t_]!r} correct={ok[t_]} | " + o_.replace("\n", " ")[:400])
        event("tool_eval", chat_val=chat_mean, per_source=res, query=qs, query_ok=ok)
        try:
            new = not tool_csv.exists()
            with open(tool_csv, "a", newline="") as fh:
                w_ = csv.writer(fh)
                if new:
                    w_.writerow(["step", "chat_val"] + list(res) + ["query_data", "ok_data", "query_100m", "ok_100m"])
                w_.writerow([step, chat_mean] + list(res.values()) + [qs["data"], ok["data"], qs["100m"], ok["100m"]])
        except Exception:
            pass

    pf = Prefetcher(make_batch, pin=DEV.type == "cuda")
    tm = dict(data=0.0, train=0.0)
    last = dict(decor=torch.zeros(()), bal=torch.zeros(()))

    # ---- rehearsal: each organ's cache is mixed in at a small scheduled rate (more when its cache loss drifts)
    def reh_rates():
        if a.rehearsal == "off":
            return {}
        R_ = S["reh"]; rates = {}
        core_oids = [o for o in model.organ_ids if o in CACHES and not CACHES[o].pinned and len(CACHES[o]) >= 16]
        for oid in model.organ_ids:
            cc = CACHES.get(oid)
            if cc is None or len(cc) < 16:
                continue
            if cc.pinned:
                r = a.rehearsal_rate * R_["boost"].get(oid, 1.0)
                if R_["refresh"].get(oid, 0) > 0:
                    r = max(r, a.rehearsal_max)
            else:
                r = a.rehearsal_core_rate / max(1, len(core_oids))
            rates[oid] = r
        tot_r = sum(rates.values())
        if tot_r > a.rehearsal_max:
            rates = {o: r * a.rehearsal_max / tot_r for o, r in rates.items()}
        return rates

    def pick_rehearsal():
        R_ = S["reh"]; R_["n_micro"] += 1
        rates = reh_rates()
        if not rates:
            return None
        R_["credit"] += sum(rates.values())
        if R_["credit"] < 1.0:
            return None
        R_["credit"] -= 1.0
        oids = list(rates); w = np.array([rates[o] for o in oids]); w = w / w.sum()
        oid = oids[int(rng.choice(len(oids), p=w))]
        cc = CACHES[oid]; Tc = min(cc.T, a.ctx)
        R_["n"] += 1
        return cc.batch(max(1, a.batch * a.ctx // Tc), DEV, T=Tc, split="train", rng=rng)

    def reh_tick():
        for oid in list(S["reh"]["refresh"]):
            S["reh"]["refresh"][oid] -= 1
            if S["reh"]["refresh"][oid] <= 0:
                del S["reh"]["refresh"][oid]
                event("rehearsal_refresh_done", organ=oid)

    @torch.no_grad()
    def cache_update():
        """emergent organs: score the latest training batch per organ and offer it to that organ's cache."""
        if "xy" not in last:
            return
        x, y = last["xy"]
        for k, oid in enumerate(model.organ_ids):
            if oid not in CACHES:
                CACHES[oid] = RehearsalCache(cap=cache_cap(a.cache_seqs, a.cache_mb, a.ctx), T=a.ctx, seed=step,
                                             provenance="main-run training text (" + ",".join(a.mix or ["train.bin"]) + ")")
            if not CACHES[oid].pinned and model.K > 1:
                cache_feed(model, k, CACHES[oid], x, y, "core", amp)
        for oid in [o for o in CACHES if o not in model.organ_ids]:
            CACHES.pop(oid)   # organ gone (its cache went into its removal bundle)

    def rehearsal_check():
        """refresher: an organ whose held-out cache loss rose above its baseline by > tol gets a burst of rehearsal."""
        rep = {}
        for k, oid in enumerate(model.organ_ids):
            cc = CACHES.get(oid)
            if cc is None or not cc.pinned or len(cc) < 16:
                continue
            lc = cache_loss(model, cc, DEV, "check", n=64, amp=amp)
            if cc.baseline is None:
                cc.baseline = lc
            drift = lc - cc.baseline
            S["reh"]["boost"][oid] = float(min(4.0, max(1.0, 1.0 + drift / max(1e-6, a.rehearsal_tol))))
            rep[oid] = dict(loss=round(lc, 4), baseline=round(cc.baseline, 4), drift=round(drift, 4), boost=round(S["reh"]["boost"][oid], 2))
            if a.rehearsal == "on" and drift > a.rehearsal_tol and oid not in S["reh"]["refresh"]:
                S["reh"]["refresh"][oid] = a.refresh_steps
                event("rehearsal_refresh", organ=oid, cache_loss=round(lc, 4), baseline=round(cc.baseline, 4), drift=round(drift, 4),
                      steps=a.refresh_steps)
        S["reh"]["last_check"] = rep
        if rep:
            event("rehearsal_check", organs=rep, rehearsal=a.rehearsal, share=round(S["reh"]["n"] / max(1, S["reh"]["n_micro"]), 4))

    def train_step():
        nonlocal step
        m, _ = lr_mult()
        for o in opts:
            for g in o.param_groups:
                g["lr"] = g["base_lr"] * m
        t_ = time.perf_counter()
        tot, n_main, reh_l = 0.0, 0, None
        for _ in range(a.accum):
            pf.step = step
            rb = pick_rehearsal()
            if rb is None:
                td = time.perf_counter(); x, y = pf.get(DEV); tm["data"] += time.perf_counter() - td
                last["xy"] = (x, y)
            else:
                x, y = rb
            stats = dict(decor=[], bal=[])
            with amp():
                loss = model(x, y, stats=stats)
                aux = 0.0
                if stats["decor"]:
                    dl = torch.stack(stats["decor"]).mean(); bl = torch.stack(stats["bal"]).mean()
                    aux = a.decor * dl + a.balance * bl
                    last["decor"], last["bal"] = dl.detach(), bl.detach()
                total = (loss + aux) / a.accum
            scaler.scale(total).backward()
            if rb is None:
                tot = tot + loss.detach(); n_main += 1
            else:
                reh_l = loss.detach()
        tot = tot / n_main if n_main else (reh_l if reh_l is not None else torch.zeros(()))
        for o in opts:
            scaler.unscale_(o)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        if not use_amp:
            ok = torch.isfinite(gn).float()
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.nan_to_num_(0.0, 0.0, 0.0).mul_(ok)
        for o in opts:
            scaler.step(o)
        scaler.update()
        for o in opts:
            o.zero_grad(set_to_none=True)
        ema.update(model)
        step += 1
        if S["reh"]["refresh"]:
            reh_tick()
        tm["train"] += time.perf_counter() - t_
        return tot, m

    # ---- generic trial: the change vs a no-change control, same steps, same schedule
    def run_trial(apply_fn, steps=None):
        nonlocal model, opts, step
        steps = steps or a.trial_steps
        snap = copy.deepcopy((model, opts, ema.sh, ema.n, scaler.state_dict()))
        s0 = step
        S["phase"], S["phase_start"] = "trial", s0
        for _ in range(steps):
            train_step()
        val_ctrl = evaluate(model, dec, amp)
        ctrl = (model, opts, ema.sh, ema.n, scaler.state_dict(), step)
        model, opts = snap[0], snap[1]; ema.sh, ema.n = snap[2], snap[3]; scaler.load_state_dict(snap[4])
        step = s0; snap = None
        info = apply_fn() or {}
        rebuild(info.get("reset", ()))
        S["phase"], S["phase_start"] = "trial", s0
        for _ in range(steps):
            train_step()
            if S["fades"]:
                update_fades()
        val_new = evaluate(model, dec, amp)
        contrib = None
        if info.get("ids"):
            contrib = evaluate(model, use_b, amp, skip=info["ids"]) - evaluate(model, use_b, amp)
        if "tol" in info:   # transplant: must not hurt the core beyond tol
            keep = val_new <= val_ctrl + info["tol"]
        else:
            keep = val_new <= val_ctrl - a.min_gain and (contrib is None or contrib >= info.get("min_contrib", a.new_min))
        if not keep:
            model, opts, ema.sh, ema.n, sc, step = ctrl
            scaler.load_state_dict(sc)
            S["fades"] = [f_ for f_ in S["fades"] if fade_cells(f_)]
        ctrl = None
        if DEV.type == "cuda":
            torch.cuda.empty_cache()
        probe_memory()
        S["phase"], S["phase_start"], S["cycle_start"] = "rewarm", step, step
        last["decor"], last["bal"] = torch.zeros(()), torch.zeros(())
        return keep, val_ctrl, val_new, contrib, info

    # ---- growth
    def cell_params():
        return sum(p.numel() for p in model.layers[0].cells[0].parameters())

    def layer_shared():
        return sum(p.numel() for n_, p in model.layers[0].named_parameters() if not n_.startswith("cells.") and n_ != "router")

    def choose_op(state):
        L, K = model.L, model.K
        pref = "deepen" if (L < a.depth_ratio * K and L + a.depth_step <= a.max_layers) else "widen"
        if S["prefer_alt"]:
            pref = "widen" if pref == "deepen" else "deepen"
        if pref == "widen" and K >= a.max_cells:
            pref = "deepen"
        if pref == "deepen" and L + a.depth_step > a.max_layers:
            pref = "widen"
        ex = lambda op: cell_params() * L if op == "widen" else (cell_params() * K + layer_shared()) * a.depth_step
        if state == "tight":   # hungry: take the cheaper growth
            pref = min(("widen", "deepen"), key=ex) if K < a.max_cells and L + a.depth_step <= a.max_layers else pref
        return pref, ex(pref)

    def mem_ok(extra):
        p_new = model.n_params() + extra
        if p_new > a.max_params:
            return False, f"params {p_new/1e6:.1f}M > cap {a.max_params/1e6:.0f}M"
        if p_new * 4 / 2**20 > a.mem_cap_mb:
            return False, f"weights {p_new*4/2**20:.0f} MB > --mem-cap-mb {a.mem_cap_mb:.0f}"
        st, f = hunger()
        if st == "near_cap":
            return False, f"hunger near_cap (headroom {f:.2f})"
        if DEV.type == "cuda" and S["vram_min_mb"] * p_new / model.n_params() > a.vram_cap_mb * 0.85:
            return False, f"VRAM even with full checkpointing would pass the cap"
        return True, ""

    def growth_decision():
        st, f = hunger()
        op, extra = choose_op(st)
        ok, why = mem_ok(extra)
        if not ok:
            event("growth_blocked", op=op, why=why); S["rejects"] += 1; return
        base, use = knockout(model, use_b[:2], amp)
        vals = sorted(use.values())
        event("usefulness", base=round(base, 4), min=round(vals[0], 4), median=round(vals[len(vals)//2], 4), max=round(vals[-1], 4),
              per_layer=[round(sum(use[(li, ci)] for ci in range(model.K)), 4) for li in range(model.L)],
              per_organ={oid: round(sum(use[(li, k)] for li in range(model.L)), 4) for k, oid in enumerate(model.organ_ids)})
        old = model.shape_str()

        def apply():
            newm = widen(model) if op == "widen" else deepen(model, a.depth_step)
            if op == "widen":   # new organ joins at alpha 0 and fades in: exactly function preserving
                with torch.no_grad():
                    for c_ in newm:
                        c_.alpha.zero_()
                start_fade("organ", model.organ_ids[-1], 0.0, 1.0, "in_done", steps=max(1, min(a.fade_steps, a.trial_steps // 3)))
            return dict(ids=new_cell_ids(model, newm))
        keep, vc, vn, contrib, info = run_trial(apply)
        event("growth_trial", op=op, frm=old, to=(model.shape_str() if keep else f"(kept {model.shape_str()})"), hunger=st,
              val_control=round(vc, 4), val_grown=round(vn, 4), new_contribution=None if contrib is None else round(contrib, 4), keep=keep)
        if keep:
            S["growths"] += 1; S["rejects"] = 0; S["prefer_alt"] = False
            save(tag=f"grow{S['growths']}_{model.shape_str()}_step{step}")
            if op == "widen" and a.export_on_growth:
                do_export("organ", model.organ_ids[-1], ema_w=False)
        else:
            S["rejects"] += 1; S["prefer_alt"] = not S["prefer_alt"]

    # ---- immune system: find freeloader cells, trial-replace them with fresh cells
    def immune():
        base, use = knockout(model, use_b[:2], amp)
        cands = [(u, li, ci) for (li, ci), u in use.items()
                 if not model.layers[li].cells[ci].protected and model.organ_ids[ci] not in model.protected_organs
                 and float(model.layers[li].cells[ci].alpha) == 1.0 and u < a.immune_rel * base]
        S["last_immune"] = step
        keep_c = []
        for u_, li_, ci_ in cands:   # a cell idle on main text but needed for its organ's own ability is not a freeloader
            cc = CACHES.get(model.organ_ids[ci_])
            if cc is not None and cc.pinned and len(cc) >= 16:
                b_c = cache_loss(model, cc, DEV, "check", n=32, amp=amp)
                u_c = cache_loss(model, cc, DEV, "check", n=32, amp=amp, skip={(li_, ci_)}) - b_c
                if u_c > a.immune_rel * b_c:
                    continue
            keep_c.append((u_, li_, ci_))
        if len(keep_c) < len(cands):
            event("immune_spared", cells=len(cands) - len(keep_c), why="useful on their organ's rehearsal cache")
        cands = keep_c
        if not cands:
            event("immune_scan", result="clean", threshold=round(a.immune_rel * base, 4), weakest=round(min(use.values()), 4)); return
        u, li, ci = min(cands)
        old_cid = model.layers[li].cells[ci].cid
        old_t = {k: v.detach().clone() for k, v in cell_tensors(model.layers[li], ci).items()}

        def apply():
            rs = reinit_cell(model, li, ci)
            start_fade("cell", model.layers[li].cells[ci].cid, 0.0, 1.0, "in_done", steps=max(1, min(a.fade_steps, a.trial_steps // 3)))
            return dict(reset=rs, ids={(li, ci)}, min_contrib=max(u, 0.0) + 1e-4)
        keep, vc, vn, _, _ = run_trial(apply)
        if keep:
            write_part(PD / "removed" / f"cell_{old_cid}_immune_step{step}", old_t,
                       dict(kind="cell", id=old_cid, layer=li, organ=model.organ_ids[ci], reason="immune: freeloader replaced",
                            usefulness=u, sig=model_sig(model), tokenizer=model.tokenizer, lineage=model.lineage, n_layers=model.L), log)
        event("immune", cell=old_cid, layer=li, organ=model.organ_ids[ci], usefulness=round(u, 5), candidates=len(cands),
              val_control=round(vc, 4), val_replaced=round(vn, 4), replaced=keep)

    def nonlocal_opts_rebuild():
        nonlocal opts
        opts = build_opts(model, a, old=opts)   # re-list params in module order (keeps resume state aligned)

    # ---- organizer: keep organs coherent (function-preserving column alignment) and report their niches
    @torch.no_grad()
    def organizer():
        S["last_org"] = step
        if model.K < 2:
            return
        x, y = dec[0]
        v_before = evaluate(model, [dec[0]], amp)
        for l in model.layers:
            l.capture = True
        evaluate(model, [dec[0]], amp)
        ps = [l.last_p.reshape(-1, model.K).float() for l in model.layers]
        for l in model.layers:
            l.capture = False; l.last_p = None
        fixed_cols = [k for k, oid in enumerate(model.organ_ids) if oid in model.protected_organs]
        moved = 0
        for li in range(1, model.L):
            A_, B_ = ps[li - 1], ps[li]
            A_ = (A_ - A_.mean(0)) / (A_.std(0) + 1e-6); B_ = (B_ - B_.mean(0)) / (B_.std(0) + 1e-6)
            C = (A_.T @ B_ / A_.shape[0]).cpu().numpy()
            perm = best_perm(C, fixed=fixed_cols)   # column k of layer li takes the cell that best matches column k below
            if perm != list(range(model.K)):
                lay = model.layers[li]
                lay.cells = nn.ModuleList([lay.cells[j] for j in perm])
                pidx = torch.tensor(perm, device=lay.router.device)
                lay.router.data = lay.router.data[pidx]
                for o in opts:
                    st_ = o.state.get(lay.router)
                    if st_:
                        for kk, vv in st_.items():
                            if torch.is_tensor(vv) and vv.dim() == 2:
                                st_[kk] = vv[pidx]
                ema.sh[lay.router] = ema.sh[lay.router][pidx]
                ps[li] = ps[li][:, pidx]
                moved += sum(1 for i, j in enumerate(perm) if i != j)
        v_after = evaluate(model, [dec[0]], amp)
        P = torch.stack(ps).mean(0)            # (N, K) organ share per token, averaged over layers
        ids = x.reshape(-1)
        niches = {}
        for k, oid in enumerate(model.organ_ids):
            sc = torch.zeros(model.vocab, device=P.device).index_add_(0, ids, P[:, k])
            cnt = torch.bincount(ids, minlength=model.vocab).float()
            rel = torch.where(cnt >= 3, sc / cnt.clamp(min=1) - 1.0 / model.K, torch.full_like(sc, -1))
            top = rel.topk(5).indices.tolist()
            niches[oid] = [base_tok.decode([i]).strip() or repr(base_tok.decode([i])) for i in top]
        load_ = {oid: round(float(P[:, k].mean()), 3) for k, oid in enumerate(model.organ_ids)}
        if moved:
            nonlocal_opts_rebuild()
        event("organizer", cells_moved=moved, function_change=round(abs(v_after - v_before), 7), load=load_, niches=niches)

    # ---- parts: export / install / remove / protect, through the inbox
    def organ_usefulness():
        base = evaluate(model, use_b[:2], amp)
        return {k: evaluate(model, use_b[:2], amp, skip={(li, k) for li in range(model.L)}) - base for k in range(model.K)}

    def do_export(kind, pid, ema_w=True):
        if ema_w:
            ema.swap(model)
        try:
            tag = f"step{step}_{model.lineage}"
            if kind == "tree":
                p = export_tree(model, PD / "exports" / f"tree_{model.shape_str()}_{tag}", dict(step=step, ema=ema_w), log)
            elif kind == "organ":
                if pid in (None, "auto"):
                    u = organ_usefulness(); k = max(u, key=u.get)
                else:
                    k = model.organ_ids.index(pid)
                if model.socket and sock is not None:
                    p = harvest_bundle(model, k, PD / "exports" / f"organ_{model.organ_ids[k]}_{tag}", sock[2],
                                       dict(note=f"exported at step {step}"), log, cache=CACHES.get(model.organ_ids[k]))
                else:
                    p = export_organ(model, k, PD / "exports" / f"organ_{model.organ_ids[k]}_{tag}", dict(step=step, ema=ema_w), log)
            elif kind == "cell":
                if pid in (None, "auto"):
                    _, use = knockout(model, use_b[:1], amp); li, ci = max(use, key=use.get)
                else:
                    li, ci = model.find_cell(pid)
                p = export_cell(model, li, ci, PD / "exports" / f"cell_{model.layers[li].cells[ci].cid}_{tag}", dict(step=step, ema=ema_w), log)
            else:
                raise ValueError(f"unknown kind {kind}")
        finally:
            if ema_w:
                ema.swap(model)
        event("part_exported", part=kind, path=str(p))
        return p

    def start_fade(kind, pid, frm, to, then, payload=None, steps=None):
        S["fades"].append(dict(kind=kind, id=pid, start=step, steps=steps or a.fade_steps, frm=frm, to=to, then=then, payload=payload))

    def fade_cells(f):
        if f["kind"] == "organ":
            if f["id"] not in model.organ_ids:
                return []
            k = model.organ_ids.index(f["id"])
            return [l.cells[k] for l in model.layers]
        loc = model.find_cell(f["id"])
        return [] if loc is None else [model.layers[loc[0]].cells[loc[1]]]

    @torch.no_grad()
    def update_fades():
        done = []
        for f in S["fades"]:
            t = min(1.0, (step - f["start"]) / max(1, f["steps"]))
            for cobj in fade_cells(f):
                cobj.alpha.fill_(f["frm"] + (f["to"] - f["frm"]) * t)
            if t >= 1.0:
                done.append(f)
        for f in done:
            S["fades"].remove(f)
            finish_fade(f)

    def finish_fade(f):
        if f["then"] == "in_done":
            if S["phase"] != "trial":   # fades inside a growth/immune trial are reported by the trial result
                event("fade_in_done", part=f["kind"], id=f["id"])
        elif f["then"] == "remove_organ" and f["id"] in model.organ_ids:
            k = model.organ_ids.index(f["id"])
            for l in model.layers:
                l.cells[k].alpha.fill_(1.0)   # archive the organ at full strength (as it was before the fade)
            if model.socket:
                p = harvest_bundle(model, k, PD / "removed" / f"organ_{f['id']}_removed_step{step}", sock[2],
                                   dict(note=f"hot-removed at step {step}"), log, cache=CACHES.get(f["id"]))
            else:
                p = export_organ(model, k, PD / "removed" / f"organ_{f['id']}_removed_step{step}", dict(step=step, reason="hot-remove"), log)
            remove_organ_column(model, k)
            CACHES.pop(f["id"], None)
            rebuild()
            event("organ_removed", id=f["id"], archived=str(p))
        elif f["then"] in ("swap_in", "reinit"):
            loc = model.find_cell(f["id"])
            if loc is None:
                return
            li, ci = loc
            model.layers[li].cells[ci].alpha.fill_(1.0)
            p = export_cell(model, li, ci, PD / "removed" / f"cell_{f['id']}_removed_step{step}", dict(step=step, reason=f["then"]), log)
            model.layers[li].cells[ci].alpha.fill_(0.0)
            if f["then"] == "reinit":
                reset = reinit_cell(model, li, ci); rebuild(reset)
                start_fade("cell", model.layers[li].cells[ci].cid, 0.0, 1.0, "in_done")
                event("cell_removed", id=f["id"], replaced_by=model.layers[li].cells[ci].cid, archived=str(p))
            else:
                man, t = read_part(f["payload"])
                reset = load_cell_weights(model, li, ci, t)
                cobj = model.layers[li].cells[ci]
                cobj.cid = man["id"] if model.find_cell(man["id"]) is None else model.new_cid()
                cobj.protected = bool(man.get("protected")); cobj.alpha.zero_()
                rebuild(reset)
                start_fade("cell", cobj.cid, 0.0, 1.0, "in_done")
                event("cell_swapped_in", old=f["id"], new=cobj.cid, layer=li, organ=model.organ_ids[ci], archived_old=str(p))

    def move_to(f, sub, reason=None):
        dst = PD / sub / f"{time.strftime('%Y%m%d_%H%M%S')}_{f.name}"
        try:
            shutil.move(str(f), str(dst))
            if reason:
                dst.with_suffix(dst.suffix + ".reason.txt").write_text(reason, encoding="utf-8")
        except Exception as e:
            log(f"could not move {f}: {e}")
        return dst

    def handle_cmd(cmd):
        c_ = cmd.get("cmd")
        pid = cmd.get("id")
        if pid == "last_installed":
            pid = S.get("last_installed")
        if c_ == "export":
            do_export(cmd.get("kind", "organ"), pid, ema_w=cmd.get("ema", True)); return None
        if c_ in ("protect", "unprotect"):
            val = c_ == "protect"
            if pid in model.organ_ids:
                (model.protected_organs.add if val else model.protected_organs.discard)(pid)
            elif model.find_cell(pid):
                li, ci = model.find_cell(pid); model.layers[li].cells[ci].protected = val
            else:
                return f"unknown id {pid}"
            event(c_, id=pid); return None
        if c_ == "remove":
            busy = {f_["id"] for f_ in S["fades"]}
            if pid in busy:
                return f"{pid} is already fading"
            if pid in model.organ_ids:
                if pid in model.protected_organs and not cmd.get("force"):
                    return f"{pid} is protected (send force: true to remove anyway)"
                if model.K < 2:
                    return "cannot remove the last organ"
                start_fade("organ", pid, 1.0, 0.0, "remove_organ"); event("remove_started", id=pid); return None
            loc = model.find_cell(pid)
            if loc:
                if model.layers[loc[0]].cells[loc[1]].protected and not cmd.get("force"):
                    return f"{pid} is protected"
                start_fade("cell", pid, 1.0, 0.0, "reinit"); event("remove_started", id=pid); return None
            return f"unknown id {pid}"
        return f"unknown command {c_!r}"

    def install(js, man, t):
        errs, warns = validate_part(model, man, t)
        st, f = hunger()
        if st == "near_cap":
            errs.append(f"hungry: near cap (headroom {f:.2f})")
        if man["kind"] == "organ" and model.n_params() + sum(v.numel() for v in t.values()) > a.max_params:
            errs.append("would pass --max-params")
        if errs:
            return "; ".join(errs)
        if man["kind"] == "organ":
            oid = install_organ(model, man, t)
            rebuild()
            start_fade("organ", oid, 0.0, 1.0, "in_done")
            S["last_installed"] = oid
            event("part_installed", part="organ", id=oid, source=man["id"], warnings=warns, fade_steps=a.fade_steps)
        else:   # cell: swap into a slot (symmetry kept): fade the target out, then the part in
            tgt = man.get("target")
            if tgt and model.find_cell(tgt):
                li, ci = model.find_cell(tgt)
            else:
                li = min(int(man.get("layer", 0)), model.L - 1)
                base = evaluate(model, use_b[:1], amp)
                us = {ci: evaluate(model, use_b[:1], amp, skip={(li, ci)}) - base for ci in range(model.K)
                      if not model.layers[li].cells[ci].protected and model.organ_ids[ci] not in model.protected_organs}
                if not us:
                    return "no unprotected cell to swap in layer"
                ci = min(us, key=us.get)
            old = model.layers[li].cells[ci].cid
            start_fade("cell", old, 1.0, 0.0, "swap_in", payload=str(js))
            S["last_installed"] = old
            event("part_install_started", part="cell", target=old, layer=li, source=man["id"], warnings=warns)
        return None

    def process_inbox():
        inbox = PD / "inbox"
        for f in sorted(inbox.glob("*.json")):
            try:
                age = time.time() - f.stat().st_mtime
                if f.name.endswith(".cmd.json"):
                    err = handle_cmd(json.loads(f.read_text(encoding="utf-8")))
                    move_to(f, "rejected" if err else "processed", err)
                    if err:
                        event("inbox_rejected", file=f.name, why=err)
                    continue
                man = json.loads(f.read_text(encoding="utf-8"))
                stp = f.with_name(man.get("file", f.with_suffix(".safetensors").name))
                if man.get("format") in ORGAN_FORMATS:
                    if not stp.exists() and age < 300:
                        continue
                    try:
                        man, t = read_bundle(f)
                        errs, _ = bundle_compat(model, man)
                        if errs:
                            raise ValueError("; ".join(errs))
                    except ValueError as e:
                        if age < 60 and "sha256" in str(e):
                            continue
                        move_to(f, "rejected", str(e))
                        if stp.exists():
                            move_to(stp, "rejected")
                        event("inbox_rejected", file=f.name, why=str(e)); continue
                    new_js = move_to(f, "processed"); move_to(stp, "processed")
                    S.setdefault("pending", []).append(dict(js=str(new_js), file=man["file"]))
                    event("transplant_queued", name=man.get("name"), version=man.get("version"))
                    continue
                if not stp.exists():
                    if age > 300:
                        move_to(f, "rejected", "safetensors file missing"); event("inbox_rejected", file=f.name, why="missing weights")
                    continue
                try:
                    man, t = read_part(f)
                except ValueError as e:
                    if age < 60:
                        continue   # maybe still being copied
                    move_to(f, "rejected", str(e)); move_to(stp, "rejected"); event("inbox_rejected", file=f.name, why=str(e)); continue
                if any(f_["kind"] == "cell" and f_["then"] == "swap_in" for f_ in S["fades"]) and man["kind"] == "cell":
                    continue   # one cell swap at a time
                new_js = move_to(f, "processed"); new_st = move_to(stp, "processed")
                (new_js.parent / man["file"]).exists() or shutil.copy2(new_st, new_js.parent / man["file"])
                err = install(new_js, man, t)
                if err:
                    move_to(new_js, "rejected", err); event("inbox_rejected", file=f.name, why=err)
            except Exception as e:
                log(f"inbox: {f.name}: {type(e).__name__}: {e}")
                move_to(f, "rejected", f"{type(e).__name__}: {e}")

    # ---- hot transplant of a portable organ bundle (control vs transplant trial)
    def transplant(item):
        js = Path(item["js"])
        man = json.loads(js.read_text(encoding="utf-8"))
        st_p = next(iter(sorted(js.parent.glob("*_" + man["file"]))), None)
        from safetensors.torch import load_file
        if st_p is None or file_sha(st_p, n=64) != man["sha256"]:
            event("transplant_failed", name=man.get("name"), why="weights missing or sha mismatch"); return
        t = load_file(str(st_p))
        rep_ = {}
        bc = RehearsalCache.from_bundle(man, t)

        def apply():
            oid, r = install_bundle(model, man, t, sock[2], align=False if a.transplant_no_align else "auto")
            rep_.update(r, oid=oid)
            if bc is not None:
                CACHES[oid] = bc
            start_fade("organ", oid, 0.0, 1.0, "in_done", steps=max(1, min(a.fade_steps, a.trial_steps // 3)))
            return dict(tol=a.transplant_tol)
        keep, vc, vn, _, _ = run_trial(apply)
        oid = rep_.get("oid")
        if keep:
            S["last_installed"] = model.organ_ids[-1]
            if oid not in CACHES:   # bundle without a cache: synthetic rehearsal sampled from the organ itself
                CACHES[oid] = synth_cache(model, model.organ_ids.index(oid), sock[2], DEV, n=128, T=min(128, a.ctx), log=log, amp=amp)
            CACHES[oid].baseline = cache_loss(model, CACHES[oid], DEV, "check", n=64, amp=amp)
        elif oid in CACHES and oid not in model.organ_ids:
            CACHES.pop(oid)
        event("transplant_trial", name=man.get("name"), version=man.get("version"), keep=keep, val_control=round(vc, 4),
              val_with_organ=round(vn, 4), tol=a.transplant_tol, fits=rep_.get("fits"), warnings=rep_.get("warnings"))

    # ---- main loop
    mp = out / "metrics.csv"
    if not mp.exists():
        with open(mp, "w", newline="") as fh:
            csv.writer(fh).writerow(["step", "minutes", "train_loss", "val_ts", "val_mix", "val_ts_ema", "val_mix_ema", "layers",
                                     "organs", "params", "lr_mult", "phase", "decor", "balance", "tok_s", "step_ms",
                                     "data_wait_ms", "ckpt_layers", "hunger", "rss_mb", "vram_mb"])
    log(f"Seed v4 FINAL start: device={DEV} precision={'fp16' if use_amp else 'fp32'} shape={model.shape_str()} "
        f"params={model.n_params():,} batch={a.batch}x{a.accum} ctx={a.ctx} mix={'on' if mix else 'off'} tokenizer={tok_info['id']} "
        f"lineage={model.lineage} parts={PD}")
    run_tl, n_tl, tok_c = 0.0, 0, 0
    if step == 0:
        event("birth", val=round(evaluate(model, dec, amp), 4))
    stop_reason = None
    try:
        while True:
            m, prog = lr_mult()
            if prog >= 1.0:
                stop_reason = "budget"; break
            if a.wall_minutes and time.time() - sess0 > a.wall_minutes * 60:
                stop_reason = "wall"; break
            if (out / "STOP").exists():
                stop_reason = "STOP file"; break
            if S["phase"] == "rewarm" and step - S["phase_start"] >= a.rewarm_steps:
                S["phase"], S["phase_start"] = "stable", step
            tl, mlt = train_step()
            if S["fades"]:
                update_fades()
            run_tl = run_tl + tl; n_tl += 1; tok_c += a.batch * a.accum * a.ctx
            if a.cache_every and step % a.cache_every == 0:
                cache_update()
            if a.rehearsal_check_every and step % a.rehearsal_check_every == 0:
                rehearsal_check()
            if step % a.inbox_every == 0 and S["phase"] in ("stable", "rewarm"):
                process_inbox()
            if step % a.eval_every == 0:
                if DEV.type == "cuda":
                    torch.cuda.synchronize()
                tr_s, dw = tm["train"], tm["data"]; tm["train"] = tm["data"] = 0.0
                vt = evaluate(model, val_ts, amp); vm = evaluate(model, val_mix, amp) if mix else float("nan")
                ema.swap(model)
                vte = evaluate(model, val_ts, amp); vme = evaluate(model, val_mix, amp) if mix else float("nan")
                ema.swap(model)
                vd = vm if (mix and a.decide_on == "mix") else vt
                S["hist"].append((step, vd)); S["hist"] = S["hist"][-200:]
                tps = tok_c / max(tr_s, 1e-6); sms = 1000 * tr_s / max(1, n_tl); dms = 1000 * dw / max(1, n_tl); tok_c = 0
                vram = torch.cuda.max_memory_reserved() / 2**20 if DEV.type == "cuda" else 0
                hs, hf = hunger()
                tlv = float(run_tl) / max(1, n_tl); ldec, lbal = float(last["decor"]), float(last["bal"])
                row = [step, round((t_off + time.time() - t0) / 60, 2), round(tlv, 4), round(vt, 4), round(vm, 4), round(vte, 4),
                       round(vme, 4), model.L, model.K, model.n_params(), round(mlt, 3), S["phase"], round(ldec, 4), round(lbal, 4),
                       int(tps), round(sms, 1), round(dms, 1), model.ckpt_layers, hs, int(rss_mb()), int(vram)]
                try:
                    with open(mp, "a", newline="") as fh:
                        csv.writer(fh).writerow(row)
                except Exception:
                    pass
                log(f"step {step} | train {tlv:.3f} | ts {vt:.3f}/{vte:.3f}ema | mix {vm:.3f}/{vme:.3f}ema | {model.shape_str()} "
                    f"{model.n_params()/1e6:.2f}M {'+'.join(model.organ_ids)} | lr x{mlt:.2f} {S['phase']} | decor {ldec:.3f} bal {lbal:.3f} | "
                    f"{int(tps):,} tok/s {sms:.0f} ms/step (data {dms:.1f}) ckpt {model.ckpt_layers}/{model.L} | {hs} | "
                    f"RSS {int(rss_mb())} MB VRAM {int(vram)} MB | reh {S['reh']['n']}/{S['reh']['n_micro']}" + (f" | fading {[f_['id'] for f_ in S['fades']]}" if S["fades"] else ""))
                run_tl, n_tl = 0.0, 0
                calm = S["phase"] == "stable" and not S["fades"]
                growing = (not fixed) and prog < 1 - a.final_decay_frac
                if calm and a.organizer_every and step - S["last_org"] >= a.organizer_every:
                    organizer()
                if calm and a.immune_every and step - S["last_immune"] >= a.immune_every and step > a.warmup:
                    immune()
                if calm and S.get("pending"):
                    st_, _ = hunger()
                    if st_ != "near_cap":
                        transplant(S["pending"].pop(0))
                since = step - S["cycle_start"]
                hs, _ = hunger()
                backoff = 2 ** min(S["rejects"], a.max_rejects) * (1.5 if hs == "tight" else 1.0)
                if growing and calm and hs != "near_cap" and since >= a.min_cycle * backoff:
                    old = [v for s_, v in S["hist"] if s_ <= step - a.plateau_window]
                    plateau = bool(old) and (old[-1] - vd) / old[-1] < a.plateau_rel
                    if plateau or (since >= a.max_cycle * backoff and hs == "well_fed"):
                        S["phase"], S["phase_start"] = "cool", step
                        event("cool_start", plateau=plateau, val=round(vd, 4), hunger=hs)
                elif S["phase"] == "cool" and step - S["phase_start"] >= a.cool_steps and not S["fades"]:
                    growth_decision()
                    S["phase"], S["phase_start"], S["cycle_start"] = "rewarm", step, step
                    if DEV.type == "cuda":
                        torch.cuda.reset_peak_memory_stats()
            if step % a.save_every == 0:
                save()
                try:
                    ema.swap(model); model.eval()
                    for pr in a.prompts.split("|")[:3]:
                        ids = torch.tensor([base_tok.encode(pr).ids], device=DEV)
                        with amp():
                            txt = base_tok.decode(model.generate(ids, 60)[0].tolist())
                        log("SAMPLE " + txt.replace("\n", " "))
                    if a.tool_eval and src_val:
                        try:
                            tool_eval()
                        except Exception as e_:
                            log(f"tool_eval failed: {e_!r}")
                finally:
                    model.train(); ema.swap(model)
    except KeyboardInterrupt:
        stop_reason = "ctrl-c"
    S["step"] = step
    save()
    log(f"stopped ({stop_reason}) at step {step} shape {model.shape_str()} organs {model.organ_ids} params {model.n_params():,}")


if __name__ == "__main__":
    main()
