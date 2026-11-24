"""tiny10m_v2 model: dense ~12M transformer for TinyStories.
Changes vs the earlier tiny10m: QK-norm, logit soft-cap, zero-init output projections,
embedding norm, fused SwiGLU (gate+up), 64-dim heads, half-split RoPE.
Tokenizer/data identical to tiny10m so losses compare directly."""
import math
from dataclasses import dataclass, asdict
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    block_size: int = 256
    n_layer: int = 14
    n_head: int = 4
    n_embd: int = 256
    mlp_hidden: int = 576
    rope_base: float = 10000.0
    softcap: float = 30.0
    tie_embeddings: bool = True


def rms(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return rms(x) * self.weight


def rope_cache(head_dim, seq_len, base):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    f = torch.outer(torch.arange(seq_len).float(), inv)
    return f.cos(), f.sin()


def apply_rope(x, cos, sin):
    d = x.size(-1) // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.nh = c.n_head
        self.hd = c.n_embd // c.n_head
        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd, bias=False)
        self.qkv.weight._muon_split = 3          # orthogonalize q,k,v separately
        self.proj = nn.Linear(c.n_embd, c.n_embd, bias=False)
        self.q_norm = RMSNorm(self.hd)
        self.k_norm = RMSNorm(self.hd)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.nh, self.hd).permute(2, 0, 3, 1, 4)
        q = apply_rope(self.q_norm(q), cos, sin)
        k = apply_rope(self.k_norm(k), cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class SwiGLU(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.h = c.mlp_hidden
        self.gate_up = nn.Linear(c.n_embd, 2 * c.mlp_hidden, bias=False)
        self.gate_up.weight._muon_split = 2
        self.down = nn.Linear(c.mlp_hidden, c.n_embd, bias=False)

    def forward(self, x):
        g, u = self.gate_up(x).split(self.h, dim=-1)
        return self.down(F.silu(g) * u)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln1 = RMSNorm(c.n_embd)
        self.attn = Attention(c)
        self.ln2 = RMSNorm(c.n_embd)
        self.mlp = SwiGLU(c)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        return x + self.mlp(self.ln2(x))


class TinyGPT2(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.cfg = c
        self.wte = nn.Embedding(c.vocab_size, c.n_embd)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layer)])
        self.ln_f = RMSNorm(c.n_embd)
        self.head = None if c.tie_embeddings else nn.Linear(c.n_embd, c.vocab_size, bias=False)
        cos, sin = rope_cache(c.n_embd // c.n_head, c.block_size, c.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.wte.weight, std=0.02)
        for n, p in self.named_parameters():
            if p.ndim == 2 and n != "wte.weight":
                if n.endswith("proj.weight") or n.endswith("down.weight"):
                    nn.init.zeros_(p)               # zero-init residual outputs
                else:
                    b = math.sqrt(3) * 0.5 / math.sqrt(p.size(1))
                    nn.init.uniform_(p, -b, b)
        if self.head is not None:
            nn.init.zeros_(self.head.weight)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = rms(self.wte(idx))
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        for blk in self.blocks:
            x = blk(x, cos, sin)
        x = self.ln_f(x)
        w = self.wte.weight if self.head is None else self.head.weight
        logits = F.linear(x, w).float()
        if self.cfg.softcap:
            logits = self.cfg.softcap * torch.tanh(logits / self.cfg.softcap)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50, eos_id=None):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.block_size:])
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            nxt = torch.multinomial(F.softmax(logits, -1), 1)
            idx = torch.cat([idx, nxt], 1)
            if eos_id is not None and nxt.item() == eos_id:
                break
        return idx


def n_params(m):
    return sum(p.numel() for p in m.parameters())
