# seed-socket-v1: the standard socket for portable Seed organs

A **socket** is the contract that lets a cell or organ trained in one Seed work in *another* Seed (a different run, seed or shape). This spec is version 1. Anything that changes the contract must get a new version string, and nothing is silently compatible across versions.

Published files (`socket/seed-socket-v1\`, all pinned by sha256 in `socket.json`):

| file | content |
|---|---|
| `socket.json` | this contract in machine-readable form, plus the sha256 of the two files below |
| `anchor_emb.safetensors` | `emb` (4096 × 384) frozen token (un)embedding, and `readout_gain` (384) frozen readout norm gain |
| `calib_tokens.safetensors` | `tokens` (16 × 128) the fixed calibration text: TinyStories-valid windows, seed 0 |

The current v1 anchor is the EMA embedding and final-norm gain of `runs/run1` at step 17550 (anchor sha `a8438384b2c1b973`).

## The contract

1. **Tokenizer**: `seed-bpe4096-v1`, sha `42ca2e21f41c629d`, vocab 4096, `<eos>` = 3.
2. **Residual width at the socket**: `d = 384`. Every socket seed has a 384-wide residual stream.
3. **Anchor space** (the key to cross-seed transfer):
   - token embedding = `anchor.emb`, **frozen**.
   - logits = `(RMSNorm(x) * anchor.readout_gain) @ anchor.emb^T`, tied and **frozen**, with logit soft-cap 30.
   - Every socket seed therefore reads tokens in, and writes predictions out, in the *same* coordinate system. This pins both ends of every seed's residual stream to one shared space.
4. **Normalization**: every cell input (attention and MLP) is `RMSNorm(x)` with gain **fixed at 1**, with no per-seed learned gain. Any scaling a cell needs is inside its own weights.
5. **Position**: interleaved complex RoPE, base 10000, applied per 32-dim head (`head_dim = 32`). Context length is the host's own; organs see positions `0..T-1` of their host.
6. **Cell interface**: a cell reads `h = RMSNorm(x)` and adds its writes to `x`. It owns everything inside:
   - `q` (n_head·32 × 384), its own K/V head `kv` (64 × 384, MQA inside the cell), and `q_norm` / `k_norm` (32 each).
   - `proj` (384 × n_head·32), SwiGLU `w12` (2·mlp × 384) and `down` (384 × mlp).
   - **Socket adapters** (rank r, default 8):
     - input: `u = h + bin(ain h)`, with `ain` r×384 and `bin` 384×r.
     - output: `o' = o + bout(aout o)`, with `aout` r×384 and `bout` 384×r.
     - `bin = bout = 0` is the identity, so a fresh adapter changes nothing. Adapters are what an adopting host tunes.
   - *Free internals*: `n_head`, `mlp` and `rank` are up to the author. A host pads a smaller organ to its own profile exactly: zero heads, zero MLP units and zero adapter ranks add nothing. A host refuses organs bigger than its profile. `head_dim` is fixed (32) because RoPE is defined per 32-dim pair set.
7. **Gating**:
   - Host cells compete through the layer router: `p = softmax(router·h)`, `gate = Σα · p`.
   - A **plug-in organ cell** has its own gate, `α · sigmoid(router_row · h)`, on both its attention and MLP writes, and stays out of the host's competition. Installing, removing or silencing an organ therefore never rescales the host's own cells.
   - `α` (fade level) = 0 means exactly absent, so every install starts as an exact no-op and fades in.
8. **Organ**: one cell per host layer (a column). It carries `depth_hints` (relative depth of each cell). A host with a different depth maps cells by nearest relative depth.
9. **Alignment at import**: rotation `R` = orthogonal Procrustes between the host's and the donor's `RMSNorm(residual)` on the calibration tokens.
   - The donor's activations ship in the bundle (`calib.L{j}`), measured **without** the organ itself, so they describe the trunk the organ expects.
   - Per layer and side: input weights `W ← W Rᵀ`, router row `r ← r Rᵀ`, `bin ← R bin`, output weights `proj/down/bout ← R ·`, `aout ← aout Rᵀ`.
   - `align=auto` (default) rotates only where Procrustes beats the identity fit by more than 0.1. The adopt command also probes identity / Procrustes / auto zero-shot and keeps the best.

## Why this design

- **Frozen shared anchor.** Two seeds trained from scratch end up with residual coordinates that are arbitrary rotations (and worse) of each other. Fixing the embedding and readout to one published matrix removes most of that freedom for free. Layer-0 inputs are *identical* across seeds, and the last layers must write in the anchor's output basis to make good predictions.
  - The middle of the network can still drift. That is what the Procrustes step is for, and it costs only a small calibration tensor in the bundle.
  - In the test, identity alignment was already the best choice between independently trained seeds. That shows the anchor does most of the work.
- **Self-contained cells** (own K/V, own norms, fixed-gain RMSNorm). A cell never depends on a per-seed shared projection it can't carry with it.
- **Low-rank adapters, zero-initialised.** Cheap (12k params per cell at r=8), exact identity at install, and enough to correct small residual mismatches during the adopter's short tune.
- **Plug-in sigmoid gate.** A competitive softmax gate is exact only at α = 0. Any non-zero organ gate rescales the host's cells, and that showed up as core-loss regressions in the box tests. An independent gate lets an organ stay quiet off-task without touching the host.

## Bundle format: `seed-organ/1.1` (reads `seed-organ/1` too)

Each bundle is:
- `NAME.safetensors`: tensors `L{j}.{q,kv,proj,w12,down,q_norm,k_norm,ain,bin,aout,bout,router}` per donor cell j, plus `calib.L{j}` (fp16 calibration activations, layer inputs and final residual).
- `NAME.json`: the manifest.
- Optional: `NAME.task_eval.jsonl` and `NAME.task_tune.jsonl`, sha-pinned in the manifest.

Manifest fields:
- **Format and contract**:
  - `format`, `socket`, `tokenizer{id,sha256}`, `anchor_sha256`, `width`.
  - `cell_profile{n_head,head_dim,mlp,rank,kv_heads}`, `gate`.
  - `n_cells`, `depth_hints`, `calib{n_seq,T}`.
- **Size and cost**: `n_params`, `cost{ram_mb, train_vram_mb_est}`.
- **Identity**: `name`, `version`, `description`, `author`, `license`, `price` (marketplace placeholder only, with no payment code).
- **Provenance**: `donor{shape,lineage,organ_id}`, `training{...}` (data summary, steps, seconds).
- **Scores**: `eval{reference_host, core_before/after, task_before/after}`.
- **Integrity**: `file`, `sha256` (64-hex of the weights), `created`.
- **Signature**: `signature{alg:"ed25519", author, public_key, sig}` over the canonical manifest JSON (sorted keys, without `signature`). The signed manifest includes the weights' sha256, so it covers the weights too. Signing uses a dependency-free pure-Python Ed25519 that passes the RFC 8032 test vector.

### Rehearsal cache (new in 1.1)

An organ carries its **hardest training material** so it can relearn in a new host and resist erosion there.
- **Tensors**, in the same safetensors file, so the weights sha256 and the signature cover them:
  - `cache.tokens`: int16 [N, T+1] token ids (vocab 4096 < 2^15).
  - `cache.maskbits`: uint8 [N, ceil(T/8)]; `np.packbits` of the target-token mask (1 = a token the organ is responsible for, e.g. answer tokens).
  - `cache.score`: fp16 [N].
  - `cache.source`: uint8 [N] (0 task, 1 core, 2 synthetic, 3 bundle, 4 harvest).
- **Size**: about 0.27 KB per 128-token sequence. The default cap is min(1024 sequences, 4 MB), so roughly 280 KB at T=128.
- **Manifest `rehearsal`**: `included`, `mode` (full | synthetic | none), `count`, `cap`, `T`, `bytes`, `sources`, `selection` (the rule below), `heldout_rule`, `tokenizer`, `data_provenance` and `data_license` (the seller declares where the data comes from), `baseline` (the donor's held-out cache loss with and without the organ), `score_range`.
  - With `included: false`, a `reason` and a `fallback` are given instead.
- **Selection**: every candidate sequence is scored by the organ itself, `score = (loss without organ - loss with organ) + 0.5 x loss with organ`, on its target tokens only. That favours where the organ matters most and what is still hard. Then:
  - drop sequences where the organ doesn't help (gap ≤ 0);
  - drop likely garbage (loss with the organ above the running 98th percentile);
  - drop near-duplicates: MinHash of token 3-grams, ≥ 0.8 similar, keep the higher score;
  - cover the range: 8 difficulty strata from running score quantiles, each with an equal quota;
  - inside a stratum, an online weighted reservoir (A-Res key `u^(1/score)`) decides who stays.

  After training, the online pool plus fresh batches are re-scored with the finished organ.
- **Held-out split**: rows with `minhash[0] % 10 == 0` (about 10%, stable by content) are never tuned on. They are the yardstick for adoption and for the refresher.
- **No-cache option**: sellers who can't share data ship `mode none` (or strip the cache later). Importers then fall back to a **synthetic** cache: the host with the organ samples from calibration-set prefixes, and the samples where the organ matters most are kept. This is weaker; see the limits in ORGANS_README.

Version rules: a host accepts only bundles whose socket, tokenizer sha, anchor sha, width and head_dim match exactly. Smaller n_head, mlp or rank is padded; larger is refused. Different depth is remapped, with a warning.
