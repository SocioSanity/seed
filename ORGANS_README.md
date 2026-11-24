# Seed organs: train an ability, share it, plug it into any Seed

The idea: everyone trains the **parts they care about** (an "organ" = a group of cells trained together on one task) on their own PC, and gets the rest from others. Organs are portable between independently trained Seeds because every socket Seed shares the standard socket (`SOCKET_SPEC.md`).

## Starter organs in this repo

- `organs/seeds/run3-trunk-final` — fluent TinyStories trunk (whole frozen seed, fp16 tree part). Chat with `python chat.py organs/seeds/run3-trunk-final.json`.
- `organs/seeds/run4-tools-reasoning-final` — tools and reasoning whole seed.
- `organs/run4-tools-reasoning-1.0.1` — run4's organ column with a rehearsal cache. Transplant with `python parts_tool.py import organs/run4-tools-reasoning-1.0.1 --host HOST.pt --out merged.pt`.

These starter organs are free. The tools in `parts_tool.py` and `seed_organs.py` are what you use to harvest, verify, and transplant organs of your own.

All commands run from the Seed folder with the CUDA Python:
```
cd .
set PY=python
```

## 0. One-time setup
- The socket already exists at `socket/seed-socket-v1`. It was made with `parts_tool.py make-socket --from runs/run1/latest.pt`. Don't remake it: every organ is tied to its anchor sha. If you share organs, share this folder too, because it is the standard.
- Optional signing key: `python parts_tool.py keygen --author yourname`. This writes `organs/keys\yourname.ed25519.key`; keep the `.key` file private. Your public key goes into every bundle you sign.

## 1. Get a host trunk (any socket Seed)
Any checkpoint trained by `seed_v4.py` (socket v1 is the default) is a host, e.g. `runs/run2/latest.pt`. Trunks can have different depth, number of cells, or random seed.

## 2. Train an organ (cheap: only the organ learns)
Task data is JSONL, one example per line: `{"prompt": "...", "answer": "..."}`. Loss and metrics are on answer tokens. You also need a held-out eval file in the same format (e.g. 256 lines). `make_task.py` shows a tiny example (the "Zorbish" made-up language).
```
python parts_tool.py train-organ --host runs/run2/latest.pt --task my_task.jsonl --task-eval my_eval.jsonl ^
    --name my-ability --version 1.0.0 --description "what it does" --author yourname --license CC-BY-4.0 ^
    --steps 1500 --ctx 128 --out organs/my\my-ability-1.0.0 --sign-key organs/keys\yourname.ed25519.key
```
What happens:
- The host trunk is **frozen**. A new plug-in column is added at α = 0, fades in over the first 10% of steps, and only the new cells (and their router rows) train.
- About 70% of the batches are your task. The other 30% are TinyStories core text, where the organ is trained to *match the trunk without it* (KL), so it learns to stay quiet off-task.
- The bundle records the trunk's core loss and your task score, before and after, in its `eval` field.
- **Rehearsal cache (default on)**: while the organ trains, and again once it's finished, the hardest and most organ-dependent task sequences are kept in a small cache inside the bundle (about 280 KB for 1024 × 128 tokens; see SOCKET_SPEC.md). Options:
  - `--cache-mode full|synthetic|none`, `--cache-seqs 1024`, `--cache-mb 4`;
  - `--data-provenance "..."` and `--data-license ...` to declare where the cached data comes from;
  - `--no-task-files`, to ship only the cache and not your tune or eval JSONL.

## 3. Harvest an existing organ (from any trained socket Seed)
```
python parts_tool.py harvest --host SEED.pt --organ o3 --out organs/my\name-1.0.0 --name ... --task t.jsonl --task-eval e.jsonl
```
Without `--organ`, the organ whose router usage is most specialised to `--task` (compared with core text) is picked.

## 4. Share: the local library
- `organs/library\NAME\VERSION\` holds the bundle files, and `organs/library\index.json` is the catalog (name, version, description, author, license, size, cost, sha, signed, scores).
- Publish: `python parts_tool.py publish BUNDLE.json`. List: `python parts_tool.py list`.
- Anything you download goes into the same layout. Always run `verify` first:
  ```
  python parts_tool.py verify BUNDLE.json      (checks the weights sha256 and the ed25519 signature)
  ```
## 5. Adopt (transplant) an organ into YOUR Seed
```
python parts_tool.py import my-ability@1.0.0 --host runs/run2/latest.pt --out organs/hosts\run2_plus_my-ability.pt --steps 400
```
Steps:
1. **Verify** the sha256 and signature, then **check compatibility**: socket, tokenizer, anchor, width, head_dim, and cell profile no bigger than the host's.
2. **Align**: try identity, Procrustes and per-layer auto, zero-shot, and keep the best.
3. **Install** as a plug-in column at α = 0. This is exact: the host's loss is unchanged at install.
4. **Adapter-only tune**: the host *and the organ's internals are frozen*. Only the organ's socket adapters and router rows train, for `--steps` steps, on the bundle's tune sample plus core text (KL to the host), while α fades in over the first third.
5. **Accept or reject** against the no-organ host:
   - core loss must not get worse by more than `--tol` (default 0.02);
   - the organ's task loss must improve by at least `--min-gain` (default 0.02) on its shipped eval set.

   Tune data, in order: the bundle's **rehearsal cache** → a shipped tune file (old 1.0 bundles) → a synthetic cache sampled from the organ itself.
   - With a cache, the accept decision uses the cache's held-out rows, so no original dataset is needed.
   - `--report-eval E.jsonl` measures an eval set for the report only.

   Accepted: a new checkpoint is written to `--out`, with the organ's cache inside it. Rejected: the bundle and a `REJECTED.json` with the reasons are **archived** in `organs/rejected\` (never deleted).

Then chat with it, or keep training it with the trainer:
```
python chat.py organs/hosts\run2_plus_my-ability.pt --prompt "..."
python chat.py runs/run2/latest.pt --parts BUNDLE.json --prompt "..."        (plug in at full strength for a quick try)
python seed_v4.py --init-from organs/hosts\run2_plus_my-ability.pt --out runs\runX ...
```

## 6. Rehearsal while the host keeps training (no more erosion)
In the trainer (`seed_v4.py`), every organ has a cache. It is saved in `latest.pt` and resumed with it.
- **Imported or task organs** rehearse their own cache on about `--rehearsal-rate` (4%) of micro-batches, as a scheduled share. This is boosted up to 4× when the organ's held-out cache loss drifts above its baseline.
- **Refresher**: if that drift passes `--rehearsal-tol` (0.1 nats), the organ gets `--refresh-steps` (100) steps at up to `--rehearsal-max` (25%). The `rehearsal_check`, `rehearsal_refresh` and `rehearsal_refresh_done` events are logged.
- **Organs that emerge in the main run** keep a cache of the main-run text they matter most for. It is updated every `--cache-every` steps and rehearsed lightly (`--rehearsal-core-rate` 1% in total).
- **Immune system**: it no longer replaces a cell that looks idle on main text but is needed on its organ's own cache.
- Hot removal and export of an organ write its cache into the bundle. `harvest` from any checkpoint exports the cache the host kept (or builds one from `--task`, or a synthetic one).
- `--rehearsal off` turns all of this off (the control).

Cache tools:
```
python parts_tool.py cache BUNDLE.json                       (view: stats + the hardest examples, decoded)
python parts_tool.py cache BUNDLE.json --action strip --out NEW_STEM          (new bundle without the cache)
python parts_tool.py cache BUNDLE.json --action rebuild --host HOST.pt [--task T.jsonl] --out NEW_STEM
python parts_tool.py eval-task --host HOST.pt --task-eval E.jsonl              (task metric + core loss)
```
Bundles are never overwritten: strip and rebuild write a new bundle (signature reset unless `--sign-key`).

## 7. Hot transplant and removal while training
- Drop `BUNDLE.json` + `BUNDLE.safetensors` into the run's `parts\inbox\`. At the next calm point the trainer runs a **control vs transplant trial**: it keeps the organ only if core val is no worse than the control + `--transplant-tol`.
- `{"cmd":"remove","id":"o3"}` fades an organ out, archives it as a bundle in `parts\removed\`, and takes it out.
- `protect` / `unprotect` work as before (see PARTS_README.md).

## What transfers well, and what doesn't (honest limits)
- **Transfers well**: abilities whose computation sits near the shared anchor, meaning token-level mappings, vocabulary and style (what to write for given input tokens), because input and output space are identical across socket Seeds.
- **Transfers less well**:
  - abilities that depend on *intermediate* features specific to the donor trunk, which another trunk computes differently. Procrustes fixes rotations, not different features.
  - organs from a much deeper or shallower donor (cells are remapped by depth).
  - organs trained on a very different data distribution than the host's.
- **Adapters are small** (rank 8). They correct offsets; they cannot re-teach the ability. The adopt step's random-organ control shows how much the tune alone can do.
- **Rehearsal cache limits**:
  - A cache holds what the organ needed *on its donor*. Hard cases that only show up in a new host aren't in it.
  - The synthetic fallback rehearses the organ's own outputs from generic prefixes, so it preserves an ability but can't improve one, and it barely triggers abilities that need special prompts.
  - Rehearsal costs a few percent of training throughput.
  - A shipped cache is real data: the author must declare its provenance and license, or ship `--cache-mode none`.
- **Core cost**: an always-on organ costs a little core loss. The plug-in gate and the KL-to-host training reduce this but do not guarantee zero, which is why adoption has a tolerance.
- **One socket only**: organs never cross socket versions, tokenizers or anchors. A different anchor means a different ecosystem.
- **Security**: a signature proves who made the bundle and that it wasn't changed. It does not prove the organ is harmless. The core-loss check is a quality gate, not a safety audit.
