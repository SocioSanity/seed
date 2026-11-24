"""tiny100m: frozen tiny10m_v2 'fluent core' + a new 768-wide tower that adds to the core's logits.

logits = softcap30( core_logits(x)  +  [0 for old ids | new_bias for new ids]  +  tower_head(tower(x)) )

* core (14x256, 12.0M) is loaded from the fluent-core checkpoint and FROZEN (requires_grad False, eval-only).
* new vocab rows for the core's tied embedding/head are trainable (init = mean of the old tokens they decompose into).
* tower: reads normalized core hidden states (layers 7 and 14) through an up-projection, plus its own token
  embedding, runs N transformer blocks at width 768, and its output head is ZERO-initialised, so at init the
  expanded model computes exactly the frozen core's function (verified by the init check)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from core_model import ModelConfig, TinyGPT2, Block, RMSNorm, rms, rope_cache

OLD_V = 8192


class Tiny100M(nn.Module):
    def __init__(self, core_cfg: dict, vocab: int, d=768, n_layer=10, n_head=12, mlp_hidden=1792, block=512, taps=(6, 13)):
        super().__init__()
        cc = dict(core_cfg); cc["block_size"] = block
        self.core = TinyGPT2(ModelConfig(**cc))
        self.core.requires_grad_(False)
        self.vocab, self.block, self.taps = vocab, block, taps
        cd = self.core.cfg.n_embd
        self.core_new = nn.Parameter(torch.zeros(vocab - OLD_V, cd))     # new rows of core tied embedding
        self.new_bias = nn.Parameter(torch.zeros(vocab - OLD_V))         # logit offset for new ids in the core path
        tc = ModelConfig(vocab_size=vocab, block_size=block, n_layer=n_layer, n_head=n_head, n_embd=d, mlp_hidden=mlp_hidden)
        self.tcfg = tc
        self.up = nn.Linear(cd * len(taps), d, bias=False)
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(tc) for _ in range(n_layer)])
        self.ln = RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        cos, sin = rope_cache(d // n_head, block, 10000.0)
        self.register_buffer("cos", cos, persistent=False); self.register_buffer("sin", sin, persistent=False)
        nn.init.normal_(self.emb.weight, std=0.02)
        for n, p in self.blocks.named_parameters():
            if p.ndim == 2:
                if n.endswith("proj.weight") or n.endswith("down.weight"): nn.init.zeros_(p)
                else: b = math.sqrt(3) * 0.5 / math.sqrt(p.size(1)); nn.init.uniform_(p, -b, b)
        b = math.sqrt(3) * 0.5 / math.sqrt(self.up.weight.size(1)); nn.init.uniform_(self.up.weight, -b, b)
        nn.init.zeros_(self.head.weight)                                  # function-preserving at init

    def train(self, mode=True):
        super().train(mode); self.core.eval(); return self

    def core_weight(self):
        return torch.cat([self.core.wte.weight, self.core_new], 0)

    def core_pass(self, idx):
        W = self.core_weight()
        with torch.no_grad():
            c = self.core
            x = rms(F.embedding(idx, W.detach()))
            T = idx.size(1); cos, sin = c.rope_cos[:T], c.rope_sin[:T]
            taps = []
            for i, blk in enumerate(c.blocks):
                x = blk(x, cos, sin)
                if i in self.taps: taps.append(rms(x))
            h = c.ln_f(x)
        logits = F.linear(h, W)                                           # grads reach core_new only
        logits = torch.cat([logits[..., :OLD_V], logits[..., OLD_V:] + self.new_bias], -1)
        return logits, torch.cat(taps, -1)

    def forward(self, idx, targets=None, core_only=False):
        logits, feats = self.core_pass(idx)
        if not core_only:
            T = idx.size(1)
            t = self.up(feats) + self.emb(idx)
            for blk in self.blocks:
                t = blk(t, self.cos[:T], self.sin[:T])
            logits = logits + self.head(self.ln(t))
        logits = 30.0 * torch.tanh(logits.float() / 30.0)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new, temperature=0.7, top_k=40, stop_ids=()):
        for _ in range(max_new):
            lg, _ = self(idx[:, -self.block:])
            lg = lg[:, -1, :] / max(temperature, 1e-6)
            if top_k:
                v, _ = torch.topk(lg, top_k); lg[lg < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(F.softmax(lg, -1), 1)
            idx = torch.cat([idx, nxt], 1)
            if nxt.item() in stop_ids: break
        return idx


def param_report(m):
    tot = sum(p.numel() for p in m.parameters())
    train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    core = sum(p.numel() for p in m.core.parameters())
    return {"total": tot, "trainable": train, "frozen_core": core, "weights_mb_fp32": tot * 4 / 2**20}
