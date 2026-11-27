"""Seed v2.0: a growing, symmetric Seed language model for the 150k Seed v2.0 tokenizer (trainer + model).

Forked from Seed v4 (read-only source). Kept VERBATIM from v4: Cell, Layer
(GQA cells + router competition + decor/balance aux), widen/deepen/reinit_cell (exact function-preserving growth),
Muon + AdamW, build_opts carry-over, _sync_opt_devices (the fused-AdamW device fix, called after every build/load/restore),
EMA, safe_save, memory probe / selective gradient checkpointing, hunger, cool -> trial-vs-control growth cycle, immune.
New in v2.0:
  * native HuggingFace `tokenizers` (150,000 ByteLevel BPE, tokenizers/seed-bpe150k-v2/tokenizer.json)
  * factorized TIED embedding/head: E (V x e_dim) shared, proj_in (e_dim -> d), proj_out (d -> e_dim);
    logits = proj_out(norm(x)) @ E^T.  E can be shared by future Seed v2.0 specialist seeds.
  * chunked cross-entropy: logits are only ever materialised for --ce-chunk tokens at a time and recomputed in
    backward (checkpointed), so a (B*T, 150000) logits tensor never exists.
  * streaming multi-source text mix (no pre-tokenisation), held-out val per source, val CE (nats/token) AND
    val BPB (bits per UTF-8 byte) on every eval, best_ce.pt / best_bpb.pt / latest.pt, stop when val_ce < --target-ce.
  * growth trials snapshot to CPU RAM (not VRAM) so the 6 GB card is never doubled up.
Dropped from v4 (not needed for a fresh English seed): socket-v1 anchor, parts inbox/hot-transplant, rehearsal caches,
organizer column alignment, tool eval, v3 data pipeline.
"""
import argparse, copy, csv, json, math, os, queue, re, shutil, sys, threading, time, uuid
os.environ.setdefault("RAYON_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

HERE = Path(__file__).resolve().parent
REPO = HERE.parent if (HERE.parent / 'tokenizers').is_dir() else HERE
ARCH = "seed-v2.0"
DEV = torch.device("cpu")
LN2 = math.log(2.0)

# ======== verbatim from seed_v4.py: model blocks ========
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



# --------------------------------------------------------------------------- Seed v2.0 model: factorized tied 150k embedding + chunked CE


def _ce_chunk(z, E, y, softcap):
    lg = F.linear(z, E.to(z.dtype)).float()
    if softcap:
        lg = softcap * torch.tanh(lg / softcap)
    return F.cross_entropy(lg, y, reduction="sum")


def _ce_chunk_tok(z, E, y, softcap):
    lg = F.linear(z, E.to(z.dtype)).float()
    if softcap:
        lg = softcap * torch.tanh(lg / softcap)
    return F.cross_entropy(lg, y, reduction="none")


class FusedChunkedCE(torch.autograd.Function):
    """Per-token CE of (z @ E^T) without ever holding more than one (chunk, V) logits block, and with a hand-written
    backward (softmax - onehot computed in place, then two matmuls): ~4 matmuls + 3 elementwise passes per chunk,
    instead of autograd's saved log_softmax + recompute. fp32 (this GPU runs fp16 GEMMs ~7x slower)."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(ctx, z, E, y, chunk):
        N = z.shape[0]
        out = torch.empty(N, device=z.device, dtype=torch.float32)
        lse = torch.empty(N, device=z.device, dtype=torch.float32)
        for s in range(0, N, chunk):
            lg = z[s:s + chunk] @ E.t()
            l_ = torch.logsumexp(lg, -1)
            lse[s:s + chunk] = l_
            out[s:s + chunk] = l_ - lg.gather(1, y[s:s + chunk, None]).squeeze(1)
            del lg
        ctx.save_for_backward(z, E, y, lse)
        ctx.chunk = chunk
        return out

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, g):
        z, E, y, lse = ctx.saved_tensors
        C = ctx.chunk
        gz = torch.empty_like(z) if ctx.needs_input_grad[0] else None
        gE = torch.zeros_like(E) if ctx.needs_input_grad[1] else None
        g = g.float()
        for s in range(0, z.shape[0], C):
            zc = z[s:s + C]
            p = zc @ E.t()
            p.sub_(lse[s:s + C, None]).exp_()
            p[torch.arange(p.shape[0], device=p.device), y[s:s + C]] -= 1.0
            p.mul_(g[s:s + C, None])
            if gz is not None:
                gz[s:s + C] = p @ E
            if gE is not None:
                gE.addmm_(p.t(), zc)
            del p
        return gz, gE, None, None


class SeedV2(nn.Module):
    def __init__(self, vocab, d, ctx, L, K, ckw, e_dim=128, softcap=30.0, ce_chunk=512):
        super().__init__()
        self.vocab, self.d, self.ctx, self.ckw, self.softcap = vocab, d, ctx, dict(ckw), softcap
        self.e_dim, self.ce_chunk = e_dim, ce_chunk
        self.E = nn.Embedding(vocab, e_dim)                       # shared factorized vocab table (tied in/out)
        nn.init.normal_(self.E.weight, std=0.02)
        self.proj_in = nn.Linear(e_dim, d, bias=False)            # e_dim -> d_model
        self.proj_out = nn.Linear(d, e_dim, bias=False)           # d_model -> e_dim (then @ E^T)
        nn.init.normal_(self.proj_in.weight, std=e_dim ** -0.5)
        nn.init.normal_(self.proj_out.weight, std=d ** -0.5)
        out_std = 0.02 / math.sqrt(2 * max(1, L))
        self.layers = nn.ModuleList([Layer(d, K, self.ckw, out_std=out_std) for _ in range(L)])
        self.nf = nn.Parameter(torch.ones(d))
        self.register_buffer("fc", rope_freqs(ctx, ckw["head_dim"], 10000.0), persistent=False)
        self.ckpt_layers = 0
        self.organ_ids = [f"o{i+1}" for i in range(K)]
        self.protected_organs = set()
        self.next_cell, self.next_organ = 1, K + 1
        self.lineage = uuid.uuid4().hex[:12]
        self.tokenizer = {}
        self.socket = None
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

    def n_vocab_params(self):
        return self.E.weight.numel()

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
        return dict(arch=ARCH, vocab=self.vocab, d=self.d, ctx=self.ctx, ckw=self.ckw, e_dim=self.e_dim, softcap=self.softcap,
                    ce_chunk=self.ce_chunk, L=self.L, K=self.K, organ_ids=list(self.organ_ids),
                    protected_organs=sorted(self.protected_organs), cids=[[c.cid for c in l.cells] for l in self.layers],
                    protected_cells=[c.cid for _, _, c in self.cells_iter() if c.protected],
                    next_cell=self.next_cell, next_organ=self.next_organ, lineage=self.lineage, tokenizer=self.tokenizer)

    @staticmethod
    def from_structure(st):
        m = SeedV2(st["vocab"], st["d"], st["ctx"], st["L"], st["K"], st["ckw"], e_dim=st["e_dim"], softcap=st["softcap"],
                   ce_chunk=st.get("ce_chunk", 512))
        m.organ_ids = list(st["organ_ids"]); m.protected_organs = set(st.get("protected_organs", []))
        pc = set(st.get("protected_cells", []))
        for layer, ids in zip(m.layers, st["cids"]):
            for c, cid in zip(layer.cells, ids):
                c.cid, c.protected = cid, cid in pc
        m.next_cell, m.next_organ, m.lineage = st["next_cell"], st["next_organ"], st["lineage"]
        m.tokenizer = st.get("tokenizer", {})
        return m

    def trunk(self, idx, skip=None, stats=None):
        x = self.proj_in(self.E(idx))
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
        return self.proj_out(x)                                    # (B, T, e_dim)

    def forward(self, idx, targets=None, skip=None, stats=None, per_token=False):
        z = self.trunk(idx, skip, stats)
        W = self.E.weight
        if targets is None:          # inference: logits for the LAST position only (B, 1, V)
            return _ce_logits(z[:, -1:], W, self.softcap)
        zf, yf = z.reshape(-1, self.e_dim), targets.reshape(-1)
        N, C = zf.shape[0], self.ce_chunk
        if not self.softcap:      # fast path (default): fused chunked CE, exact, no (N, V) tensor ever
            lt = FusedChunkedCE.apply(zf, W, yf, C)
            return lt if per_token else lt.sum() / N
        ck = self.training and torch.is_grad_enabled()
        fn = _ce_chunk_tok if per_token else _ce_chunk
        outs = []
        for s in range(0, N, C):
            if ck:
                outs.append(torch.utils.checkpoint.checkpoint(fn, zf[s:s + C], W, yf[s:s + C], self.softcap, use_reentrant=False))
            else:
                outs.append(fn(zf[s:s + C], W, yf[s:s + C], self.softcap))
        if per_token:
            return torch.cat(outs)
        return torch.stack(outs).sum() / N

    @torch.no_grad()
    def generate(self, idx, n, temp=0.8, top_k=40, top_p=0.95, rep_pen=1.1, stop_id=None):
        for _ in range(n):
            logits = self(idx[:, -self.ctx:])[:, -1, :].float()
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


def _ce_logits(z, W, softcap):
    lg = F.linear(z, W.to(z.dtype)).float()
    return softcap * torch.tanh(lg / softcap) if softcap else lg


# ======== verbatim from seed_v4.py: structural ops, optimisers, EMA, safe saving ========
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
        elif n in ("emb.weight", "head", "E.weight", "proj_in.weight", "proj_out.weight"):
            adam_decay.append(p)
        else:
            adam_nodecay.append(p)
    return muon, adam_decay, adam_nodecay


def _sync_opt_devices(opt):
    """Put every optimiser state tensor on its param's device and keep the AdamW `fused` flag correct.
    Fused AdamW (used on CUDA) requires `step` AND the moments to live on the param device; a plain
    foreach AdamW keeps `step` on CPU. load_state_dict() restores the checkpoint's param_groups (so a
    CPU-saved, fused=False checkpoint silently disables fused on GPU and leaves `step` on CPU), and
    build_opts() carries state over by clone() which preserves the old device. After growth that mixes a
    CPU `step` with CUDA params inside one fused step -> RuntimeError. Call this after every load/migrate."""
    want_fused = (DEV.type == "cuda")
    for g in opt.param_groups:
        is_adamw = "fused" in g
        if is_adamw:
            g["fused"] = want_fused
        for q in g["params"]:
            st = opt.state.get(q)
            if not st:
                continue
            for k, v in list(st.items()):
                if not torch.is_tensor(v):
                    continue
                if is_adamw and k == "step":
                    st[k] = v.to(device=q.device, dtype=torch.float32) if want_fused else v.detach().to("cpu")
                elif v.device != q.device:
                    st[k] = v.to(q.device)


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
    _sync_opt_devices(muon); _sync_opt_devices(adam)
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


# ======== verbatim from seed_v4.py: eval / misc ========
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


# --------------------------------------------------------------------------- data: streaming multi-source text mix (native HF tokenizers)


def load_tokenizer(path):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(path))
    return tok


def token_byte_lengths(tok, eos_bytes=2):
    """UTF-8 bytes each token stands for (ByteLevel BPE: one vocab char = one byte). Specials = 0 (eos = the '\\n\\n' it replaces)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\u00a1"), ord("\u00ac") + 1)) + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    inv = {chr(c): b for b, c in zip(bs, cs)}
    V = tok.get_vocab_size()
    out = np.zeros(V, dtype=np.int64)
    try:
        special = {t.content for t in tok.get_added_tokens_decoder().values()}
    except Exception:
        special = set()
    for s, i in tok.get_vocab(with_added_tokens=True).items():
        if i >= V:
            continue
        out[i] = 0 if s in special else sum(1 for ch in s if ch in inv)
    eid = tok.token_to_id("<eos>")
    if eid is not None:
        out[eid] = eos_bytes
    return out


class Source:
    def __init__(self, spec):
        self.name, self.path, self.weight = spec["name"], spec["path"], float(spec["weight"])
        self.sep = spec.get("sep", "blank")
        size = os.path.getsize(self.path)
        if spec.get("val_path"):
            self.tr = (self.path, 0, size)
            self.va = (spec["val_path"], 0, os.path.getsize(spec["val_path"]))
        else:
            cut = int(size * (1 - float(spec.get("holdout_frac", 0.01))))
            self.tr = (self.path, 0, cut)
            self.va = (self.path, cut, size)
        self.bpt = 4.5           # running estimate of bytes per token, sizes each read
        self.license = spec.get("license", "")
        self.max_digit_frac = float(spec.get("max_digit_frac", 0.0))


class TextMix:
    """Each row: one source drawn by weight, a random read from that source's train (or held-out) region, aligned to a
    document start, documents joined by <eos>, tokenised by the 150k tokenizer, cut to ctx+1 tokens."""

    def __init__(self, specs, tok, ctx):
        self.src = [Source(s) for s in specs]
        w = np.array([s.weight for s in self.src], dtype=np.float64)
        self.p = w / w.sum()
        self.tok, self.ctx = tok, ctx
        self.eos = tok.token_to_id("<eos>")
        self.fh = {}

    def _f(self, p):
        if p not in self.fh:
            self.fh[p] = open(p, "rb")
        return self.fh[p]

    def _text(self, s, region, rng, nbytes):
        for _ in range(6):   # optional quality gate: skip digit-dump chunks (e.g. "pi to 1M places" e-texts)
            t = self._text1(s, region, rng, nbytes)
            if not s.max_digit_frac or sum(ch.isdigit() for ch in t) <= s.max_digit_frac * max(1, len(t)):
                return t
        return t

    def _text1(self, s, region, rng, nbytes):
        p, lo, hi = region
        off = int(rng.integers(lo, max(lo + 1, hi - nbytes - 1)))
        f = self._f(p); f.seek(off)
        t = f.read(min(nbytes, hi - off)).decode("utf-8", "ignore")
        t = t.replace("\r\n", "\n").replace("<|endoftext|>", "<eos>")
        if s.sep == "blank":
            t = re.sub(r"\n[ \t]*\n+", "<eos>", t)
        j = t.find("<eos>")
        if 0 <= j < len(t) // 2:
            t = t[j:]                                   # start exactly at a document boundary
        else:
            k = t.find("\n")
            t = t[k + 1:] if 0 <= k < len(t) // 2 else t
        return t

    def rows(self, which, rng, held=False):
        T = self.ctx + 1
        texts, need = [], []
        for k in which:
            s = self.src[k]
            texts.append(self._text(s, s.va if held else s.tr, rng, int(T * s.bpt * 1.35) + 64))
        encs = self.tok.encode_batch(texts, add_special_tokens=False)
        out = np.zeros((len(which), T), dtype=np.int64)
        for r, (k, e, t) in enumerate(zip(which, encs, texts)):
            ids = e.ids
            s = self.src[k]
            tries = 0
            while len(ids) < T and tries < 6:
                tries += 1
                t2 = self._text(s, s.va if held else s.tr, rng, int(T * s.bpt * 1.35 * (tries + 1)) + 64)
                ids = ids + [self.eos] + self.tok.encode(t2, add_special_tokens=False).ids
            if len(ids) < T:
                ids = ids + [self.eos] * (T - len(ids))
            if len(t) > 256 and len(e.ids) > 32:
                s.bpt = 0.98 * s.bpt + 0.02 * (len(t.encode("utf-8")) / len(e.ids))
            out[r] = ids[:T]
        return torch.from_numpy(out)

    def batch(self, B, rng):
        which = rng.choice(len(self.src), size=B, p=self.p)
        t = self.rows(list(which), rng)
        return t[:, :-1].contiguous(), t[:, 1:].contiguous()

    def val_set(self, n_total, seed=991, min_per=8):
        """Fixed held-out sequences, allocated by mix weight (at least min_per per source). Returns (tokens, src_ids)."""
        rng = np.random.default_rng(seed)
        n = [max(min_per, int(round(n_total * p))) for p in self.p]
        toks, sid = [], []
        for k, nk in enumerate(n):
            for i in range(0, nk, 16):
                toks.append(self.rows([k] * min(16, nk - i), rng, held=True)); sid += [k] * min(16, nk - i)
        toks = torch.cat(toks)
        perm = torch.from_numpy(np.random.default_rng(seed + 1).permutation(len(toks)))
        return toks[perm], torch.tensor(sid)[perm]


# --------------------------------------------------------------------------- trainer


def get_args(argv=None):
    ap = argparse.ArgumentParser(description="Seed v2.0 trainer (dynamic growth only)")
    A = ap.add_argument
    A("--tokenizer", default=str(REPO / "tokenizers" / "seed-bpe150k-v2" / "tokenizer.json"))
    A("--mix-json", default=str(HERE / "mix.example.json"), help="list of {name,path,weight,val_path|holdout_frac,sep,license}")
    A("--out", default=str(REPO / "runs" / "v2-eng1")); A("--fallback-dir", default=None)
    A("--device", default="cuda"); A("--threads", type=int, default=2)
    A("--priority", default="below_normal", choices=["idle", "below_normal", "normal"])
    A("--minutes", type=float, default=1e7, help="LR-schedule horizon; huge = no final decay, train until --target-ce")
    A("--max-steps", type=int, default=0); A("--wall-minutes", type=float, default=0)
    A("--resume", action="store_true"); A("--seed", type=int, default=1234)
    # shape (start small; growth does the rest)
    A("--d", type=int, default=512); A("--ctx", type=int, default=512)
    A("--n-head", type=int, default=4); A("--head-dim", type=int, default=64); A("--mlp", type=int, default=512)
    A("--e-dim", type=int, default=128, help="factorized tied vocab table width (150000 x e_dim)")
    A("--ce-chunk", type=int, default=512, help="tokens per chunk of the checkpointed logits+CE")
    A("--start-layers", type=int, default=4); A("--start-cells", type=int, default=2); A("--softcap", type=float, default=0.0, help="0 = fused chunked CE (fast); >0 = v4 tanh softcap via checkpointed chunks (~40%% slower)")
    # optimisation
    A("--batch", type=int, default=8); A("--accum", type=int, default=4)
    A("--muon-lr", type=float, default=0.01); A("--adam-lr", type=float, default=1.5e-3)
    A("--muon-wd", type=float, default=0.01); A("--wd", type=float, default=0.05)
    A("--warmup", type=int, default=500); A("--final-decay-frac", type=float, default=0.15)
    A("--final-lr-frac", type=float, default=0.05); A("--clip", type=float, default=1.0); A("--ema", type=float, default=0.999)
    A("--amp", default="auto", choices=["auto", "fp16", "off"])
    A("--grad-ckpt", default="auto", choices=["auto", "on", "off"])
    A("--decor", type=float, default=0.05); A("--balance", type=float, default=0.01)
    # growth cycle (v4 defaults)
    A("--min-cycle", type=int, default=1500); A("--max-cycle", type=int, default=5000)
    A("--plateau-window", type=int, default=600); A("--plateau-rel", type=float, default=0.01)
    A("--cool-steps", type=int, default=300); A("--cool-frac", type=float, default=0.25)
    A("--rewarm-steps", type=int, default=300); A("--rewarm-floor", type=float, default=0.2)
    A("--trial-steps", type=int, default=300); A("--min-gain", type=float, default=0.0); A("--new-min", type=float, default=0.001)
    A("--depth-ratio", type=float, default=4.0); A("--depth-step", type=int, default=2)
    A("--max-layers", type=int, default=24); A("--max-cells", type=int, default=6)
    A("--max-params", type=float, default=150e6); A("--max-rejects", type=int, default=3)
    A("--mem-cap-mb", type=float, default=2048, help="cap on MODEL WEIGHTS (fp32 bytes)")
    A("--vram-cap-mb", type=float, default=4800, help="whole-process GPU cap")
    A("--rss-warn-mb", type=float, default=10000); A("--min-sys-free-mb", type=float, default=1024)
    A("--hunger-fed", type=float, default=0.5); A("--hunger-tight", type=float, default=0.15)
    A("--immune-every", type=int, default=6000); A("--immune-rel", type=float, default=0.002)
    A("--fade-steps", type=int, default=300)
    # eval / logging / stop
    A("--val-seqs", type=int, default=96); A("--eval-bs", type=int, default=8)
    A("--eval-every", type=int, default=250); A("--save-every", type=int, default=1500); A("--best-every", type=int, default=250)
    A("--target-ce", type=float, default=1.0, help="stop when mixed val CE (nats/token, raw or EMA) < this. BPB is NOT a stop condition")
    A("--prompts", default="Once upon a time|The history of the city|User: Why is the sky blue?\nAssistant:")
    A("--sample-tokens", type=int, default=80)
    return ap.parse_args(argv)


def lower_priority(level):
    try:
        import psutil
        p = psutil.Process()
        if os.name == "nt":
            p.nice(psutil.IDLE_PRIORITY_CLASS if level == "idle" else psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            p.nice(19 if level == "idle" else 10)
    except Exception:
        pass


def cpu_tree(o):
    if torch.is_tensor(o):
        return o.detach().to("cpu", copy=True)
    if isinstance(o, dict):
        return {k: cpu_tree(v) for k, v in o.items()}
    if isinstance(o, list):
        return [cpu_tree(v) for v in o]
    if isinstance(o, tuple):
        return tuple(cpu_tree(v) for v in o)
    return copy.deepcopy(o)


def main():
    a = get_args()
    global DEV
    DEV = torch.device("cuda" if a.device == "cuda" and torch.cuda.is_available() else "cpu")
    torch.set_num_threads(a.threads); torch.set_num_interop_threads(1)
    if a.priority != "normal":
        lower_priority(a.priority)
    if DEV.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.vram_cap_mb / (torch.cuda.get_device_properties(0).total_memory / 2**20)))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "removed").mkdir(exist_ok=True)
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

    # ---- tokenizer + data
    tok = load_tokenizer(a.tokenizer)
    V = tok.get_vocab_size(); eos_id = tok.token_to_id("<eos>")
    tok_sha = file_sha16(a.tokenizer)
    tok_info = dict(path=str(a.tokenizer), sha256_16=tok_sha, vocab=V, eos_id=eos_id)
    specs = json.loads(Path(a.mix_json).read_text(encoding="utf-8"))
    mix = TextMix(specs, tok, a.ctx)
    tbytes = torch.from_numpy(token_byte_lengths(tok))
    vtok, vsrc = mix.val_set(a.val_seqs)
    vx, vy = vtok[:, :-1].contiguous(), vtok[:, 1:].contiguous()
    vbytes = tbytes[vy]                                             # (N, T) bytes per target token
    dec = [(vx[i:i + a.eval_bs].to(DEV), vy[i:i + a.eval_bs].to(DEV)) for i in range(0, len(vx), a.eval_bs)]
    use_b = dec[:4]
    log(f"tokenizer {a.tokenizer} vocab {V:,} eos {eos_id} sha {tok_sha} | mix: " +
        ", ".join(f"{s.name} w={p:.3f}" for s, p in zip(mix.src, mix.p)) +
        f" | val {len(vx)} seqs x {a.ctx} = {vy.numel():,} tokens, {int(vbytes.sum()):,} bytes")

    @torch.no_grad()
    def eval_full():
        """Per-token CE over the whole held-out set -> mixed val_ce (nats/token), val_bpb (bits/byte), per-source both."""
        was = model.training; model.eval()
        nll = torch.zeros(len(vx), dtype=torch.float64)
        for bi, (x, y) in enumerate(dec):
            with amp():
                l = model(x, y, per_token=True).view(x.shape[0], -1).double().sum(1)
            nll[bi * a.eval_bs: bi * a.eval_bs + x.shape[0]] = l.cpu()
        model.train(was)
        ntok = torch.full((len(vx),), vy.shape[1], dtype=torch.float64); nb = vbytes.sum(1).double()
        res = dict(ce=float(nll.sum() / ntok.sum()), bpb=float(nll.sum() / LN2 / nb.sum()), src={})
        for k, s in enumerate(mix.src):
            m_ = vsrc == k
            if m_.any():
                res["src"][s.name] = (round(float(nll[m_].sum() / ntok[m_].sum()), 4), round(float(nll[m_].sum() / LN2 / nb[m_].sum()), 4))
        return res

    # ---- model (always dynamic growth: no fixed shape exists in v2.0)
    ckw = dict(n_head=a.n_head, head_dim=a.head_dim, mlp=a.mlp)
    ck = out / "latest.pt"
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    S = dict(step=0, cycle_start=0, phase="stable", phase_start=0, rejects=0, prefer_alt=False, hist=[], t_offset=0.0,
             growths=0, fades=[], last_immune=0, vram_min_mb=0.0, best_ce=float("inf"), best_bpb=float("inf"),
             best_ce_step=-10**9, best_bpb_step=-10**9, tokens=0)
    c = None
    if a.resume and ck.exists():
        c = torch.load(ck, map_location="cpu", weights_only=False)
        model = SeedV2.from_structure(c["structure"])
    else:
        model = SeedV2(V, a.d, a.ctx, a.start_layers, a.start_cells, ckw, e_dim=a.e_dim, softcap=a.softcap, ce_chunk=a.ce_chunk)
    model.tokenizer = tok_info
    model = model.to(DEV)
    opts = build_opts(model, a)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = EMA(model, a.ema)
    if c is not None:
        model.load_state_dict(c["model"])
        for o, sd in zip(opts, c["opts"]):
            o.load_state_dict(sd)
            _sync_opt_devices(o)
        for o, base in zip(opts, (a.muon_lr, a.adam_lr)):
            for g in o.param_groups:
                g["base_lr"] = base
        if c.get("scaler"):
            scaler.load_state_dict(c["scaler"])
        ema.load_by_name(model, c["ema"]); ema.n = c.get("ema_n", 0)
        S.update(c["S"])
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
    TOK_STEP = a.batch * a.accum * a.ctx

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
                with torch.enable_grad():
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
        n_state = sum(p.numel() for p in model.parameters()) * 4 * 3
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

    def hunger():
        w_mb = model.n_params() * 4 / 2**20
        r_head = 1 - w_mb / a.mem_cap_mb
        r = rss_mb()
        if r == r and r > a.rss_warn_mb:
            r_head = min(r_head, a.hunger_tight - 0.01 if r > 1.25 * a.rss_warn_mb else (a.hunger_fed + a.hunger_tight) / 2)
        if sys_avail_mb() < a.min_sys_free_mb:
            r_head = min(r_head, 0.1)
        v_head = 1 - S["vram_min_mb"] / a.vram_cap_mb if DEV.type == "cuda" else 1.0
        f = min(r_head, v_head)
        return ("well_fed" if f > a.hunger_fed else "tight" if f > a.hunger_tight else "near_cap"), f

    def save():
        S["step"] = step; S["t_offset"] = t_off + (time.time() - t0)
        obj = dict(structure=model.structure(), model=model.state_dict(), opts=[o.state_dict() for o in opts],
                   scaler=scaler.state_dict(), ema=ema.state_by_name(model), ema_n=ema.n, S=S, args=vars(a),
                   np_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state())
        return safe_save(obj, ck, log, a.fallback_dir)

    def save_best(name, metrics):
        obj = dict(structure=model.structure(), model=model.state_dict(), ema=ema.state_by_name(model), step=step,
                   metrics=metrics, args=vars(a), tokenizer=tok_info)
        return safe_save(obj, out / name, log, a.fallback_dir, retries=2)

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

    pf = Prefetcher(lambda st: mix.batch(a.batch, rng), pin=DEV.type == "cuda")
    tm = dict(data=0.0, train=0.0)
    last = dict(decor=torch.zeros(()), bal=torch.zeros(()))

    def train_step():
        nonlocal step
        m, _ = lr_mult()
        for o in opts:
            for g in o.param_groups:
                g["lr"] = g["base_lr"] * m
        t_ = time.perf_counter()
        tot = 0.0
        for _ in range(a.accum):
            td = time.perf_counter(); x, y = pf.get(DEV); tm["data"] += time.perf_counter() - td
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
            tot = tot + loss.detach()
        tot = tot / a.accum
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
        S["tokens"] += TOK_STEP
        tm["train"] += time.perf_counter() - t_
        return tot, m

    def safe_train_step():
        try:
            return train_step()
        except torch.cuda.OutOfMemoryError:
            for o in opts:
                o.zero_grad(set_to_none=True)
            model.zero_grad(set_to_none=True); torch.cuda.empty_cache()
            model.ckpt_layers = model.L
            log(f"WARNING: CUDA OOM at step {step}: step skipped, checkpointing all {model.L} layers")
            return None, lr_mult()[0]

    # ---- trial vs control, snapshots in CPU RAM
    def snapshot():
        return dict(structure=model.structure(), model=cpu_tree(model.state_dict()), opts=[cpu_tree(o.state_dict()) for o in opts],
                    ema=cpu_tree(ema.state_by_name(model)), ema_n=ema.n, scaler=scaler.state_dict(),
                    fades=copy.deepcopy(S["fades"]), step=step, tokens=S["tokens"])

    def restore(sn):
        nonlocal model, opts, step
        model = None; opts = None
        if DEV.type == "cuda":
            torch.cuda.empty_cache()
        m_ = SeedV2.from_structure(sn["structure"])
        m_.load_state_dict(sn["model"]); model = m_.to(DEV); model.train()
        opts = build_opts(model, a)
        for o, sd in zip(opts, sn["opts"]):
            o.load_state_dict(sd); _sync_opt_devices(o)
        for o, base in zip(opts, (a.muon_lr, a.adam_lr)):
            for g in o.param_groups:
                g["base_lr"] = base
        ema.sync(model); ema.load_by_name(model, sn["ema"]); ema.n = sn["ema_n"]
        scaler.load_state_dict(sn["scaler"]); S["fades"] = sn["fades"]; step = sn["step"]; S["tokens"] = sn["tokens"]

    def run_trial(apply_fn, steps=None):
        nonlocal model, opts, step
        steps = steps or a.trial_steps
        snap = snapshot()
        s0 = step
        S["phase"], S["phase_start"] = "trial", s0
        for _ in range(steps):
            safe_train_step()
            if S["fades"]:
                update_fades()
        val_ctrl = evaluate(model, dec, amp)
        ctrl = snapshot()
        restore(snap); snap = None
        info = apply_fn() or {}
        rebuild(info.get("reset", ()))
        S["phase"], S["phase_start"] = "trial", s0
        for _ in range(steps):
            safe_train_step()
            if S["fades"]:
                update_fades()
        val_new = evaluate(model, dec, amp)
        contrib = None
        if info.get("ids"):
            contrib = evaluate(model, use_b, amp, skip=info["ids"]) - evaluate(model, use_b, amp)
        keep = val_new <= val_ctrl - a.min_gain and (contrib is None or contrib >= info.get("min_contrib", a.new_min))
        if not keep:
            restore(ctrl)
        ctrl = None
        if DEV.type == "cuda":
            torch.cuda.empty_cache()
        probe_memory()
        S["phase"], S["phase_start"], S["cycle_start"] = "rewarm", step, step
        last["decor"], last["bal"] = torch.zeros(()), torch.zeros(())
        return keep, val_ctrl, val_new, contrib, info

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
        if state == "tight":
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
            return False, "VRAM even with full checkpointing would pass the cap"
        return True, ""

    def start_fade(kind, pid, frm, to, then, steps=None):
        S["fades"].append(dict(kind=kind, id=pid, start=step, steps=steps or a.fade_steps, frm=frm, to=to, then=then))

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
            if S["phase"] != "trial":
                event("fade_in_done", part=f["kind"], id=f["id"])

    def growth_decision():
        st, f = hunger()
        op, extra = choose_op(st)
        ok, why = mem_ok(extra)
        if not ok:
            event("growth_blocked", op=op, why=why); S["rejects"] += 1; return
        old = model.shape_str()

        def apply():
            newm = widen(model) if op == "widen" else deepen(model, a.depth_step)
            if op == "widen":
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
            save()
        else:
            S["rejects"] += 1; S["prefer_alt"] = not S["prefer_alt"]

    def immune():
        base, use = knockout(model, use_b[:2], amp)
        cands = [(u, li, ci) for (li, ci), u in use.items()
                 if not model.layers[li].cells[ci].protected and float(model.layers[li].cells[ci].alpha) == 1.0
                 and u < a.immune_rel * base]
        S["last_immune"] = step
        if not cands:
            event("immune_scan", result="clean", threshold=round(a.immune_rel * base, 4), weakest=round(min(use.values()), 4)); return
        u, li, ci = min(cands)
        old_cid = model.layers[li].cells[ci].cid
        old_t = {k: v.detach().cpu().clone() for k, v in model.layers[li].cells[ci].state_dict().items()}

        def apply():
            rs = reinit_cell(model, li, ci)
            start_fade("cell", model.layers[li].cells[ci].cid, 0.0, 1.0, "in_done", steps=max(1, min(a.fade_steps, a.trial_steps // 3)))
            return dict(reset=rs, ids={(li, ci)}, min_contrib=max(u, 0.0) + 1e-4)
        keep, vc, vn, _, _ = run_trial(apply)
        if keep:   # archive, never delete
            safe_save(dict(cid=old_cid, layer=li, organ=model.organ_ids[ci], usefulness=u, tensors=old_t),
                      out / "removed" / f"cell_{old_cid}_immune_step{step}.pt", log, None, retries=2)
        event("immune", cell=old_cid, layer=li, usefulness=round(u, 5), val_control=round(vc, 4), val_replaced=round(vn, 4), replaced=keep)

    def samples():
        res = []
        try:
            ema.swap(model); model.eval()
            for pr in a.prompts.split("|"):
                pr = pr.replace("\\n", "\n")
                ids = torch.tensor([tok.encode(pr, add_special_tokens=False).ids], device=DEV)
                with amp():
                    txt = tok.decode(model.generate(ids, a.sample_tokens, stop_id=eos_id)[0].tolist(), skip_special_tokens=False)
                res.append(txt)
                log("SAMPLE " + txt.replace("\n", " \\n "))
        finally:
            model.train(); ema.swap(model)
        try:
            with open(out / "samples.txt", "a", encoding="utf-8") as fh:
                fh.write(f"===== step {step} shape {model.shape_str()} params {model.n_params():,}\n" + "\n---\n".join(res) + "\n\n")
        except Exception:
            pass

    mp = out / "metrics.csv"
    if not mp.exists():
        with open(mp, "w", newline="") as fh:
            csv.writer(fh).writerow(["step", "minutes", "tokens", "train_loss", "val_ce", "val_bpb", "val_ce_ema", "val_bpb_ema",
                                     "per_source_ema", "layers", "organs", "params", "lr_mult", "phase", "decor", "balance",
                                     "tok_s", "step_ms", "data_wait_ms", "ckpt_layers", "hunger", "rss_mb", "vram_peak_mb"])
    log(f"Seed v2.0 start: device={DEV} {torch.cuda.get_device_name(0) if DEV.type == 'cuda' else ''} precision={'fp16' if use_amp else 'fp32'} "
        f"shape={model.shape_str()} params={model.n_params():,} (vocab table {model.n_vocab_params():,} = {V}x{a.e_dim}, "
        f"trunk {model.n_params() - model.n_vocab_params():,}) d={a.d} cell=({a.n_head}h x {a.head_dim} + mlp {a.mlp}) "
        f"batch={a.batch}x{a.accum}x{a.ctx}={TOK_STEP:,} tok/step ce_chunk={a.ce_chunk} growth cap {a.max_layers}L x {a.max_cells}K "
        f"<= {a.max_params/1e6:.0f}M params | target val_ce < {a.target_ce} | lineage {model.lineage}")
    if step == 0:
        r0 = eval_full()
        event("birth", val_ce=round(r0["ce"], 4), val_bpb=round(r0["bpb"], 4))
    run_tl, n_tl, tok_c = 0.0, 0, 0
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
            tl, mlt = safe_train_step()
            if tl is None:
                continue
            if S["fades"]:
                update_fades()
            run_tl = run_tl + tl; n_tl += 1; tok_c += TOK_STEP
            last_step = a.max_steps and step >= a.max_steps
            if step % a.eval_every == 0 or last_step:
                if DEV.type == "cuda":
                    torch.cuda.synchronize()
                tr_s, dw = tm["train"], tm["data"]; tm["train"] = tm["data"] = 0.0
                r = eval_full()
                ema.swap(model); re_ = eval_full(); ema.swap(model)
                vd = r["ce"]
                S["hist"].append((step, vd)); S["hist"] = S["hist"][-200:]
                tps = tok_c / max(tr_s, 1e-6); sms = 1000 * tr_s / max(1, n_tl); dms = 1000 * dw / max(1, n_tl); tok_c = 0
                vram = torch.cuda.max_memory_allocated() / 2**20 if DEV.type == "cuda" else 0
                vres = torch.cuda.max_memory_reserved() / 2**20 if DEV.type == "cuda" else 0
                hs, hf = hunger()
                tlv = float(run_tl) / max(1, n_tl); ldec, lbal = float(last["decor"]), float(last["bal"])
                row = [step, round((t_off + time.time() - t0) / 60, 2), S["tokens"], round(tlv, 4), round(r["ce"], 4), round(r["bpb"], 4),
                       round(re_["ce"], 4), round(re_["bpb"], 4), json.dumps(re_["src"]), model.L, model.K, model.n_params(),
                       round(mlt, 3), S["phase"], round(ldec, 4), round(lbal, 4), int(tps), round(sms, 1), round(dms, 1),
                       model.ckpt_layers, hs, int(rss_mb()), int(vram)]
                try:
                    with open(mp, "a", newline="") as fh:
                        csv.writer(fh).writerow(row)
                except Exception:
                    pass
                log(f"step {step} | train {tlv:.3f} | val_ce {r['ce']:.4f} val_bpb {r['bpb']:.4f} | EMA val_ce {re_['ce']:.4f} "
                    f"val_bpb {re_['bpb']:.4f} | {model.shape_str()} {model.n_params()/1e6:.2f}M | lr x{mlt:.2f} {S['phase']} | "
                    f"{int(tps):,} tok/s {sms:.0f} ms/step (data {dms:.1f}) ckpt {model.ckpt_layers}/{model.L} | {hs} | "
                    f"VRAM peak {int(vram)} MB (reserved {int(vres)}) RSS {int(rss_mb())} MB | tokens {S['tokens']/1e6:.1f}M"
                    + (f" | fading {[f_['id'] for f_ in S['fades']]}" if S["fades"] else ""))
                log("  per-source EMA (ce, bpb): " + ", ".join(f"{k} {v[0]:.3f}/{v[1]:.3f}" for k, v in re_["src"].items()))
                run_tl, n_tl = 0.0, 0
                best_ce, best_bpb = min(r["ce"], re_["ce"]), min(r["bpb"], re_["bpb"])
                use_ema = re_["ce"] <= r["ce"]
                metrics = dict(step=step, val_ce=r["ce"], val_bpb=r["bpb"], val_ce_ema=re_["ce"], val_bpb_ema=re_["bpb"],
                               per_source_ema=re_["src"], shape=model.shape_str(), params=model.n_params(), best_is_ema=use_ema)
                if best_ce < S["best_ce"] and (step - S["best_ce_step"] >= a.best_every or best_ce < a.target_ce):
                    S["best_ce"], S["best_ce_step"] = best_ce, step; save_best("best_ce.pt", metrics)
                    log(f"  new best_ce {best_ce:.4f} -> best_ce.pt")
                if best_bpb < S["best_bpb"] and (step - S["best_bpb_step"] >= a.best_every or best_ce < a.target_ce):
                    S["best_bpb"], S["best_bpb_step"] = best_bpb, step; save_best("best_bpb.pt", metrics)
                if best_ce < a.target_ce:
                    event("target_reached", val_ce=round(best_ce, 4), val_bpb=round(best_bpb, 4), target=a.target_ce)
                    stop_reason = f"TARGET val_ce {best_ce:.4f} < {a.target_ce}"
                    samples(); break
                if DEV.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                calm = S["phase"] == "stable" and not S["fades"]
                growing = prog < 1 - a.final_decay_frac and not a.max_steps
                if calm and a.immune_every and step - S["last_immune"] >= a.immune_every and step > a.warmup:
                    immune()
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
            if step % a.save_every == 0 or last_step:
                save()
                samples()
            if last_step:
                stop_reason = "max-steps"; break
    except KeyboardInterrupt:
        stop_reason = "ctrl-c"
    S["step"] = step
    save()
    log(f"stopped ({stop_reason}) at step {step} shape {model.shape_str()} params {model.n_params():,}")


def file_sha16(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


if __name__ == "__main__":
    main()
