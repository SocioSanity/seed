"""Seed organs: portable abilities over seed-socket-v1.

Flows (all used by parts_tool.py):
  make_socket   publish the socket (anchor + calibration set) once
  train_organ   train a NEW organ on a task, attached to a FROZEN host trunk (only the organ learns)
  harvest       export an organ column of any socket seed as a bundle (picked by router usage on a task)
  adopt         import a bundle into ANOTHER seed: compatibility -> Procrustes alignment -> fade-in ->
                adapter-only tune -> accept/reject against the no-organ host (rejects archived)
  library       local shop folder with index.json (publish / list / resolve)
  sign/verify   ed25519 signature of the canonical manifest (pure Python, no dependencies)
"""
import copy, json, math, os, random, shutil, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
import torch.nn.functional as F
import seed_v4 as S
import ed25519_min as ed

HERE = Path(__file__).resolve().parent
LIB_DEFAULT = HERE / "organs" / "library"
REJ_DEFAULT = HERE / "organs" / "rejected"
KEYS_DEFAULT = HERE / "organs" / "keys"
SEP = "<|endoftext|>"


def log(s):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)

# ---------------------------------------------------------------- data


class Core:
    """TinyStories token files (the shared 'core' distribution every seed is checked on)."""

    def __init__(self, data_dir):
        self.train = np.memmap(Path(data_dir) / "train.bin", dtype=np.uint16, mode="r")
        self.valid = np.memmap(Path(data_dir) / "valid.bin", dtype=np.uint16, mode="r")

    def batch(self, B, T, rng, dev):
        s = rng.integers(0, len(self.train) - T - 1, size=B)
        x = torch.tensor(np.stack([self.train[i:i + T + 1] for i in s]).astype(np.int64))
        return x[:, :-1].to(dev), x[:, 1:].to(dev)

    def val(self, T, dev, n=12, B=16, seed=999):
        r = np.random.default_rng(seed); out = []
        for _ in range(n):
            s = r.integers(0, len(self.valid) - T - 1, size=B)
            x = torch.tensor(np.stack([self.valid[i:i + T + 1] for i in s]).astype(np.int64))
            out.append((x[:, :-1].to(dev), x[:, 1:].to(dev)))
        return out


class Task:
    """Task data = JSONL lines {"prompt": ..., "answer": ...}. Loss/metrics only on answer tokens."""

    def __init__(self, path, tok):
        self.rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        self.eos = tok.token_to_id("<eos>")
        self.enc = [(tok.encode(r["prompt"]).ids, tok.encode(r["answer"]).ids + [self.eos]) for r in self.rows]

    def batch(self, B, T, rng, dev):
        X = torch.full((B, T + 1), self.eos, dtype=torch.long); Y = torch.full((B, T), -100, dtype=torch.long)
        for b in range(B):
            seq, lab = [self.eos], [-100]
            while len(seq) < T + 1:
                p, a = self.enc[int(rng.integers(len(self.enc)))]
                seq += p + a; lab += [-100] * len(p) + a
            seq, lab = seq[:T + 1], lab[:T + 1]
            X[b] = torch.tensor(seq)
            Y[b] = torch.tensor(lab[1:])
        return X[:, :-1].to(dev), Y.to(dev)

    @torch.no_grad()
    def metric(self, model, dev, n=256):
        """answer-token loss, answer-token accuracy (teacher forced), exact match of whole answers."""
        model.eval()
        tot_l, tot_n, tot_c, ex = 0.0, 0, 0, 0
        items = self.enc[:n]
        for i in range(0, len(items), 32):
            chunk = items[i:i + 32]
            T = max(len(p) + len(a) for p, a in chunk) + 1
            X = torch.full((len(chunk), T), self.eos, dtype=torch.long); Y = torch.full((len(chunk), T), -100, dtype=torch.long)
            for b, (p, a) in enumerate(chunk):
                s = [self.eos] + p + a
                X[b, :len(s)] = torch.tensor(s)
                Y[b, len(p):len(s) - 1] = torch.tensor(a)
            X, Y = X.to(dev), Y.to(dev)
            lg = model(X)
            m = Y != -100
            l = F.cross_entropy(lg[m], Y[m], reduction="sum")
            ok = (lg.argmax(-1) == Y) & m
            tot_l += float(l); tot_n += int(m.sum()); tot_c += int(ok.sum())
            ex += int(((ok | ~m).all(1)).sum())
        model.train()
        return dict(task_loss=round(tot_l / tot_n, 4), token_acc=round(tot_c / tot_n, 4), exact=round(ex / len(items), 4), n=len(items))


def eval_core(model, batches):
    return S.evaluate(model, batches, S.nullctx)

# ---------------------------------------------------------------- keys / signatures


def keygen(name, keys_dir=KEYS_DEFAULT):
    keys_dir = Path(keys_dir); keys_dir.mkdir(parents=True, exist_ok=True)
    kp = keys_dir / f"{name}.ed25519.key"
    if kp.exists():
        raise SystemExit(f"{kp} exists (not overwriting a key)")
    seed = os.urandom(32)
    kp.write_text(seed.hex(), encoding="utf-8")
    pub = ed.public_key(seed).hex()
    (keys_dir / f"{name}.ed25519.pub").write_text(pub, encoding="utf-8")
    return kp, pub


def make_signer(key_path, author):
    seed = bytes.fromhex(Path(key_path).read_text().strip())
    pub = ed.public_key(seed).hex()
    return lambda payload: dict(alg="ed25519", author=author, public_key=pub, sig=ed.sign(seed, payload).hex(),
                                signed="canonical manifest JSON (sorted keys, no 'signature'), includes the weights sha256")


def sign_bundle(js, signer):
    js = Path(js); man = json.loads(js.read_text(encoding="utf-8"))
    man["signature"] = signer(S.bundle_sha_payload(man))
    js.write_text(json.dumps(man, indent=1), encoding="utf-8")
    return man["signature"]


def verify_bundle(js, trusted=None):
    """sha256 of weights + ed25519 signature (if present). Returns dict."""
    man, t = S.read_bundle(js)   # raises on sha mismatch
    res = dict(sha256_ok=True, signed=False, signature_ok=None, trusted=None)
    sg = man.get("signature")
    if sg:
        res["signed"] = True
        res["signature_ok"] = ed.verify(bytes.fromhex(sg["public_key"]), S.bundle_sha_payload(man), bytes.fromhex(sg["sig"]))
        if trusted is not None:
            res["trusted"] = sg["public_key"] in trusted
    return man, t, res

# ---------------------------------------------------------------- helpers


def load_host(path, dev):
    m, info = S.load_model_any(path, use_ema=True, device=dev)
    if not m.socket:
        raise SystemExit(f"{path} is not a seed-socket model")
    return m, info


def save_host(model, path, extra=None, caches=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    obj = dict(structure=model.structure(), model=sd, ema={n: p.detach().cpu().float() for n, p in model.named_parameters()},
               S=dict(step=0), extra=extra or {}, caches={oid: c.state() for oid, c in (caches or {}).items()})
    S.safe_save(obj, path, print)
    return path


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad_(False)


def row_mask_hooks(model, k):
    """Only router row k gets gradient (the organ's own gate)."""
    hs = []
    for l in model.layers:
        l.router.requires_grad_(True)
        msk = torch.zeros_like(l.router); msk[k] = 1
        hs.append(l.router.register_hook(lambda g, msk=msk: g * msk))
    return hs


def set_alpha(model, k, a):
    with torch.no_grad():
        for l in model.layers:
            l.cells[k].alpha.fill_(a)


def wsd(step, n, warm=50, decay=0.2):
    if step < warm:
        return (step + 1) / warm
    if step > n * (1 - decay):
        return max(0.05, (n - step) / (n * decay))
    return 1.0


def run_steps(model, params_muon, params_adam, batch_fn, n, k, lr_m, lr_a, fade_frac, log_every=100, distill=True):
    opt_m = S.Muon(params_muon, lr=lr_m, momentum=0.95) if params_muon else None
    opt_a = torch.optim.AdamW(params_adam, lr=lr_a, betas=(0.9, 0.95), weight_decay=0.0)
    model.train()
    hist = []
    for st in range(n):
        set_alpha(model, k, min(1.0, (st + 1) / max(1, int(n * fade_frac))))
        m = wsd(st, n)
        for o, base in ((opt_m, lr_m), (opt_a, lr_a)):
            if o:
                for g in o.param_groups:
                    g["lr"] = base * m
        x, y, kind = batch_fn(st)
        if kind == "core" and distill:
            # core text: match the host WITHOUT the organ (KL), so the organ learns to stay quiet off-task
            with torch.no_grad():
                tl = model(x, skip={(li, k) for li in range(model.L)}).float()
            sl = model(x).float()
            loss = F.kl_div(F.log_softmax(sl, -1), F.log_softmax(tl, -1), log_target=True, reduction="none").sum(-1).mean()
        else:
            loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params_muon + params_adam, 1.0)
        for o in (opt_m, opt_a):
            if o:
                o.step(); o.zero_grad(set_to_none=True)
        hist.append(float(loss.detach()))
        if (st + 1) % log_every == 0:
            log(f"  step {st+1}/{n} loss {np.mean(hist[-log_every:]):.4f}")
    set_alpha(model, k, 1.0)
    return float(np.mean(hist[-50:]))

# ---------------------------------------------------------------- train-organ


def build_task_cache(model, k, task, ctx, dev, rng, cap, n_batches=48, batch=32, pool=None, provenance="", data_license=""):
    """final selection: re-score the online pool + fresh task batches with the FINISHED organ; keep the hardest, diverse set."""
    cache = S.RehearsalCache(cap=cap, T=ctx, pinned=True, provenance=provenance, data_license=data_license, seed=1)
    if pool is not None and len(pool):
        for i in range(0, len(pool), 64):
            x, y = pool.rows(torch.arange(i, min(i + 64, len(pool))), dev, ctx)
            S.cache_feed(model, k, cache, x, y, "task")
    for _ in range(n_batches):
        x, y = task.batch(batch, ctx, rng, dev)
        S.cache_feed(model, k, cache, x, y, "task")
    return cache


def train_organ(host_path, task_path, eval_path, out_stem, data_dir, socket_dir, steps=1500, batch=32, ctx=128,
                task_frac=0.7, lr_m=0.02, lr_a=3e-3, meta=None, dev="cuda", signer=None, save_host_to=None, seed=0,
                cache_mode="full", cache_seqs=1024, cache_mb=4.0, provenance=None, data_license=None, ship_files=True):
    spec, anchor, calib = S.load_socket(socket_dir)
    model, info = load_host(host_path, dev)
    tok = S.find_tokenizer(model)
    task, ev = Task(task_path, tok), Task(eval_path, tok)
    core = Core(data_dir); cval = core.val(min(ctx, model.ctx), dev)
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    c0, t0 = eval_core(model, cval), ev.metric(model, dev)
    log(f"host {model.shape_str()} before organ: core {c0:.4f} task {t0}")
    freeze_all(model)
    new = S.widen(model)                     # fresh column, zero outputs (exact no-op), alpha fades in
    k = model.K - 1
    for c in new:                            # trained as a plug-in from the start (same gating it will have in any host)
        c.to(dev); c.plugin = True
    with torch.no_grad():
        for l in model.layers:
            l.router.data[k].zero_()
    model.to(dev)
    pm, pa = [], []
    for c in new:
        for n_, p in c.named_parameters():
            if n_ in S.ADAPTER_KEYS:
                continue
            p.requires_grad_(True)
            (pm if p.dim() == 2 else pa).append(p)
    hooks = row_mask_hooks(model, k)
    pa += [l.router for l in model.layers]
    log(f"training organ {model.organ_ids[k]}: {sum(p.numel() for c in new for p in c.parameters() if p.requires_grad):,} organ weights "
        f"(+ its router rows) train; host trunk frozen ({model.n_params():,} total)")

    cap = S.cache_cap(cache_seqs, cache_mb, ctx)
    prov = provenance or f"seller task file {Path(task_path).name} ({len(task.rows)} examples)"
    pool = S.RehearsalCache(cap=cap, T=ctx, pinned=True, provenance=prov, data_license=data_license or "") if cache_mode == "full" else None
    n_task = [0]

    def bf(st):
        if rng.random() < task_frac:
            x, y = task.batch(batch, ctx, rng, dev)
            n_task[0] += 1
            if pool is not None and st > 0.3 * steps and n_task[0] % 4 == 0:   # online: hardest examples while it learns
                S.cache_feed(model, k, pool, x, y, "task")
            return x, y, "task"
        return (*core.batch(batch, ctx, rng, dev), "core")
    t_start = time.time()
    final_loss = run_steps(model, pm, pa, bf, steps, k, lr_m, lr_a, fade_frac=0.1)
    for h in hooks:
        h.remove()
    secs = time.time() - t_start
    c1, t1 = eval_core(model, cval), ev.metric(model, dev)
    log(f"with organ: core {c1:.4f} (delta {c1-c0:+.4f}) task {t1}  [{secs:.0f}s]")
    meta = dict(meta or {})
    meta.setdefault("training", dict(task_file=Path(task_path).name, task_examples=len(task.rows), steps=steps, batch=batch, ctx=ctx,
                                     task_frac=task_frac, core="TinyStoriesV2 train.bin", seconds=round(secs), final_loss=round(final_loss, 4),
                                     host=dict(path=str(host_path), shape=model.shape_str(), lineage=model.lineage, frozen=True)))
    meta.setdefault("eval", dict(reference_host=dict(lineage=model.lineage, shape=f"{model.L}x{model.K-1}"),
                                 core_before=round(c0, 4), core_after=round(c1, 4), task_before=t0, task_after=t1,
                                 task_eval_file=Path(eval_path).name))
    t_c = time.time()
    if cache_mode == "full":
        cache = build_task_cache(model, k, task, ctx, dev, rng, cap, pool=pool, provenance=prov, data_license=data_license or "")
    elif cache_mode == "synthetic":
        cache = S.synth_cache(model, k, calib, dev, n=min(cap, 512), T=ctx, cap=cap, log=log)
        cache.provenance = "synthetic: sampled from the organ on the donor (no seller data)"
    else:
        cache = None
    if cache is not None:
        cache.baseline = S.cache_loss(model, cache, dev, "check")
        wo = S.cache_loss(model, cache, dev, "check", skip_k=k)
        log(f"rehearsal cache: {cache.stats()} | held-out loss with organ {cache.baseline:.4f} without {wo:.4f} [{time.time()-t_c:.0f}s]")
        rh = S.cache_manifest(cache, model.tokenizer, mode=cache_mode)
        rh["baseline"] = dict(heldout_loss_with_organ=round(cache.baseline, 4), heldout_loss_without_organ=round(wo, 4),
                              note="measured on the donor host")
    else:
        rh = S.cache_manifest(None, model.tokenizer, included=False)
    js = S.harvest_bundle(model, k, out_stem, calib, meta, log, cache=cache, rehearsal=rh)
    if ship_files:
        ship_task_files(js, task_path, eval_path)
    if signer:
        sign_bundle(js, signer)
    if save_host_to:
        save_host(model, save_host_to, dict(organ=model.organ_ids[k]), caches={model.organ_ids[k]: cache} if cache else None)
    return js, dict(core_before=c0, core_after=c1, task_before=t0, task_after=t1, seconds=secs,
                    cache=None if cache is None else cache.stats())


def ship_task_files(js, task_path, eval_path, n_tune=512):
    """Ship the eval set and a small tune sample beside the bundle (named in the manifest, sha-pinned)."""
    js = Path(js); man = json.loads(js.read_text(encoding="utf-8"))
    stem = js.with_suffix("")
    tune_p = Path(str(stem) + ".task_tune.jsonl"); ev_p = Path(str(stem) + ".task_eval.jsonl")
    rows = [l for l in open(task_path, encoding="utf-8") if l.strip()]
    random.Random(0).shuffle(rows)
    tune_p.write_text("".join(rows[:n_tune]), encoding="utf-8")
    shutil.copy2(eval_path, ev_p)
    man["task_files"] = dict(tune=dict(file=tune_p.name, sha256=S.file_sha(tune_p, 64)), eval=dict(file=ev_p.name, sha256=S.file_sha(ev_p, 64)))
    man["signature"] = None   # any edit invalidates an old signature; re-sign after
    js.write_text(json.dumps(man, indent=1), encoding="utf-8")
    return man

# ---------------------------------------------------------------- harvest


@torch.no_grad()
def organ_usage(model, batches):
    for l in model.layers:
        l.capture = True
    for x, y in batches:
        model(x)
    u = torch.stack([l.last_p.float().mean((0, 1)) for l in model.layers]).mean(0)   # last batch is enough
    for l in model.layers:
        l.capture = False; l.last_p = None
    return u


def harvest(host_path, out_stem, socket_dir, organ=None, task_path=None, data_dir=None, meta=None, dev="cuda", signer=None,
            cache_mode="full", cache_seqs=1024, cache_mb=4.0):
    spec, anchor, calib = S.load_socket(socket_dir)
    model, info = load_host(host_path, dev)
    if organ:
        k = model.organ_ids.index(organ); why = "named"
    elif task_path and model.K > 1:
        tok = S.find_tokenizer(model); rng = np.random.default_rng(0)
        tb = [Task(task_path, tok).batch(16, 128, rng, dev) for _ in range(2)]
        cb = Core(data_dir).val(128, dev, n=2)
        spec_ = organ_usage(model, tb) - organ_usage(model, cb)
        k = int(spec_.argmax()); why = f"router specialisation on the task {spec_.tolist()}"
    else:
        k = model.K - 1; why = "last organ"
    log(f"harvesting {model.organ_ids[k]} from {model.shape_str()} ({why})")
    oid = model.organ_ids[k]
    cache = S.load_caches(host_path).get(oid) if cache_mode != "none" else None
    how = "kept by the host (formed with the organ)"
    if cache is None and cache_mode != "none":
        ctx = min(128, model.ctx); cap = S.cache_cap(cache_seqs, cache_mb, ctx)
        if task_path:
            tok = S.find_tokenizer(model)
            cache = build_task_cache(model, k, Task(task_path, tok), ctx, dev, np.random.default_rng(0), cap,
                                     provenance=f"task file {Path(task_path).name}")
            how = "built at harvest from the task file"
        else:
            cache = S.synth_cache(model, k, calib, dev, n=min(cap, 256), T=ctx, cap=cap, log=log); how = "synthetic at harvest"
    if cache is not None and len(cache):
        cache.baseline = S.cache_loss(model, cache, dev, "check")
        log(f"cache for {oid}: {len(cache)} seqs, {cache.nbytes()/1024:.0f} KB ({how})")
    rh = S.cache_manifest(cache, model.tokenizer, included=cache is not None and len(cache) > 0, mode=cache_mode, note=how)
    return S.harvest_bundle(model, k, out_stem, calib, meta, log, signer, cache=cache, rehearsal=rh)

# ---------------------------------------------------------------- adopt


def archive_reject(js, reason, rej_dir=REJ_DEFAULT):
    js = Path(js); d = Path(rej_dir) / f"{time.strftime('%Y%m%d_%H%M%S')}_{js.stem}"
    d.mkdir(parents=True, exist_ok=True)
    for f in js.parent.glob(js.stem + "*"):
        shutil.copy2(f, d / f.name)
    (d / "REJECTED.json").write_text(json.dumps(reason, indent=1, default=str), encoding="utf-8")
    return d


def adopt(host_path, bundle_js, out_path, data_dir, socket_dir, steps=300, batch=32, ctx=128, tol=0.02, min_gain=0.02,
          align="auto", tune_task_frac=0.5, lr=3e-3, dev="cuda", control=None, seed=0, rej_dir=REJ_DEFAULT, trusted=None,
          report_eval=None, use_task_files=True):
    """Tune data, in order: the bundle's rehearsal cache (1.1) -> its shipped tune file (1) -> synthetic rehearsal.
    report_eval: an extra eval JSONL used ONLY for reporting (never for tuning or for the accept decision)."""
    """control: None (real adoption) | 'random' (same-shape random organ, same tune: is the gain really the organ's?)."""
    spec, anchor, calib = S.load_socket(socket_dir)
    try:
        man, t, ver = verify_bundle(bundle_js, trusted)
    except ValueError as e:
        r = dict(stage="integrity", errors=[str(e)], bundle=str(bundle_js))
        return dict(accepted=False, reason=r, archived=str(archive_reject(bundle_js, r, rej_dir)))
    host, info = load_host(host_path, dev)
    errs, warns = S.bundle_compat(host, man)
    if ver["signed"] and not ver["signature_ok"]:
        errs = ["signature does not match the manifest (edited after signing)"] + errs
    if errs:
        r = dict(stage="compatibility", errors=errs, bundle=str(bundle_js), host=str(host_path))
        d = archive_reject(bundle_js, r, rej_dir)
        return dict(accepted=False, reason=r, archived=str(d))
    tok = S.find_tokenizer(host)
    tf = man.get("task_files", {})
    bdir = Path(bundle_js).parent
    ev = Task(bdir / tf["eval"]["file"], tok) if (tf.get("eval") and use_task_files) else None
    tune = Task(bdir / tf["tune"]["file"], tok) if (tf.get("tune") and use_task_files) else None
    rev = Task(report_eval, tok) if report_eval else None
    bcache = S.RehearsalCache.from_bundle(man, t)
    tune_src = "rehearsal cache" if bcache is not None else ("shipped tune file" if tune else "synthetic (no cache, no tune file)")
    for kk in ("eval", "tune"):
        if tf.get(kk) and S.file_sha(bdir / tf[kk]["file"], 64) != tf[kk]["sha256"]:
            raise ValueError(f"task {kk} file sha mismatch")
    core = Core(data_dir); cval = core.val(min(ctx, host.ctx), dev)
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    c0 = eval_core(host, cval); t0 = ev.metric(host, dev) if ev else None
    r0 = rev.metric(host, dev) if rev else None
    h0 = S.cache_loss(host, bcache, dev, "check", T=min(ctx, host.ctx)) if bcache is not None else None   # host without the organ
    if control == "random":   # same shapes, random weights (outputs small random), to test that gains come from the organ
        g = torch.Generator().manual_seed(seed)
        t = {kk: (torch.randn(v.shape, generator=g) * (0.02 if kk.endswith(("proj", "down")) else v.float().std().clamp_min(1e-3))
                  if v.dim() == 2 and not kk.startswith(("calib.", "cache.")) and not kk.endswith(("bin", "bout")) else v) for kk, v in t.items()}
    probe = {}
    if align == "auto":   # pick identity vs Procrustes by a zero-shot probe (organ at full strength, no tuning)
        pb = cval[:4]
        names = {False: "identity", True: "procrustes", "auto": "auto-per-layer"}
        for cand in (False, True, "auto"):
            hc = copy.deepcopy(host)
            oc, _ = S.install_bundle(hc, man, t, calib, align=cand)
            hc.to(dev); set_alpha(hc, hc.organ_ids.index(oc), 1.0)
            score = eval_core(hc, pb) + (ev.metric(hc, dev, n=64)["task_loss"] if ev else
                                         (S.cache_loss(hc, bcache, dev, "check", n=64, T=min(ctx, hc.ctx)) if bcache is not None else 0.0))
            probe[names[cand]] = round(score, 4)
            del hc
        best = min(probe, key=probe.get)
        align = {v: k for k, v in names.items()}[best]
        log(f"alignment probe (core + task loss, zero-shot, organ at full strength): {probe} -> {best}")
    oid, rep = S.install_bundle(host, man, t, calib, align=align)
    host.to(dev)
    k = host.organ_ids.index(oid)
    c_inst = eval_core(host, cval)
    freeze_all(host)
    pa = []
    for l in host.layers:
        c = l.cells[k]
        for kk in S.ADAPTER_KEYS:
            getattr(c, kk).requires_grad_(True); pa.append(getattr(c, kk))
    hooks = row_mask_hooks(host, k)
    pa += [l.router for l in host.layers]
    if bcache is None and tune is None:   # nothing shipped: rehearse the organ's own samples
        set_alpha(host, k, 1.0)
        bcache = S.synth_cache(host, k, calib, dev, n=256, T=min(ctx, host.ctx), log=log)
        set_alpha(host, k, 0.0)
        h0 = S.cache_loss(host, bcache, dev, "check", skip_k=k)
    Tc = min(ctx, host.ctx, bcache.T) if bcache is not None else ctx
    bs_c = max(1, batch * ctx // Tc) if bcache is not None else batch

    def bf(st):
        if rng.random() < tune_task_frac:
            if bcache is not None:
                return (*bcache.batch(bs_c, dev, T=Tc, split="train", rng=rng), "task")
            if tune:
                return (*tune.batch(batch, ctx, rng, dev), "task")
        return (*core.batch(batch, ctx, rng, dev), "core")
    t_s = time.time()
    run_steps(host, [], pa, bf, steps, k, 0.0, lr, fade_frac=1 / 3)
    for h in hooks:
        h.remove()
    c1 = eval_core(host, cval); t1 = ev.metric(host, dev) if ev else None
    r1 = rev.metric(host, dev) if rev else None
    h1 = S.cache_loss(host, bcache, dev, "check", T=Tc) if bcache is not None else None
    if ev:
        gain, accept_on = t0["task_loss"] - t1["task_loss"], "shipped task eval"
    elif bcache is not None:
        gain, accept_on = h0 - h1, "rehearsal cache held-out split (never tuned on)"
    else:
        gain, accept_on = None, "core only"
    ok_core = c1 - c0 <= tol
    ok_task = gain is None or gain >= min_gain
    res = dict(accepted=bool(ok_core and ok_task), organ=oid, name=man.get("name"), version=man.get("version"), aligned=str(align),
               control=control, core_before=round(c0, 4), core_after_install_alpha0=round(c_inst, 4), core_after=round(c1, 4),
               core_delta=round(c1 - c0, 4), tol=tol, task_before=t0, task_after=t1, task_loss_gain=None if gain is None else round(gain, 4),
               min_gain=min_gain, tune_steps=steps, tune_seconds=round(time.time() - t_s), verify=ver, warnings=rep["warnings"],
               layer_map=rep["layer_map"], fits=rep["fits"], alignment_probe=probe, accept_on=accept_on, tune_source=tune_src,
               cache=None if bcache is None else dict(bcache.stats(), heldout_loss_host_before=None if h0 is None else round(h0, 4),
                                                      heldout_loss_after=None if h1 is None else round(h1, 4)),
               report_eval=None if rev is None else dict(file=str(report_eval), before=r0, after=r1,
                                                         note="reporting only: not used for tuning or the decision"))
    if bcache is not None:
        bcache.baseline = h1   # the refresher's reference in this host
        bcache.pinned = True
    if res["accepted"] and out_path and control is None:
        res["saved"] = str(save_host(host, out_path, dict(adopted=res), caches={oid: bcache} if bcache is not None else None))
    elif not res["accepted"]:
        why = [] if ok_core else [f"core regressed {c1-c0:+.4f} > tol {tol}"]
        why += [] if ok_task else [f"task gain {gain:+.4f} < {min_gain}"]
        res["reason"] = why
        if control is None:
            res["archived"] = str(archive_reject(bundle_js, res, rej_dir))
    return res

# ---------------------------------------------------------------- library (local shop)


def lib_index(lib=LIB_DEFAULT):
    p = Path(lib) / "index.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else dict(format="seed-organ-library/1", organs=[])


def publish(bundle_js, lib=LIB_DEFAULT):
    man, t, ver = verify_bundle(bundle_js)
    lib = Path(lib); d = lib / man["name"] / man["version"]
    d.mkdir(parents=True, exist_ok=True)
    src = Path(bundle_js)
    for f in src.parent.glob(src.stem + "*"):
        shutil.copy2(f, d / f.name)
    idx = lib_index(lib)
    idx["organs"] = [o for o in idx["organs"] if not (o["name"] == man["name"] and o["version"] == man["version"])]
    ev = man.get("eval", {})
    idx["organs"].append(dict(name=man["name"], version=man["version"], description=man.get("description"), author=man.get("author"),
                              license=man.get("license"), price=man.get("price"), socket=man["socket"], width=man["width"],
                              n_params=man["n_params"], cost=man.get("cost"), sha256=man["sha256"], signed=ver["signed"],
                              task_after=ev.get("task_after"), core_delta=None if "core_after" not in ev else round(ev["core_after"] - ev["core_before"], 4),
                              path=str((d / src.name).relative_to(lib)), added=time.strftime("%Y-%m-%d %H:%M:%S")))
    idx["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = lib / "index.json.tmp"; tmp.write_text(json.dumps(idx, indent=1), encoding="utf-8"); os.replace(tmp, lib / "index.json")
    return d / src.name


def resolve(name_or_path, lib=LIB_DEFAULT):
    p = Path(name_or_path)
    if p.suffix == ".json" and p.exists():
        return p
    name, _, ver = str(name_or_path).partition("@")
    cands = [o for o in lib_index(lib)["organs"] if o["name"] == name and (not ver or o["version"] == ver)]
    if not cands:
        raise SystemExit(f"{name_or_path} not in {lib}\\index.json")
    return Path(lib) / sorted(cands, key=lambda o: o["version"])[-1]["path"]

# ---------------------------------------------------------------- rehearsal cache tools (parts_tool cache ...)


def rewrite_bundle(src_js, out_stem, tensors, man_updates, signer=None):
    """Write a NEW bundle (never overwrites the source): same manifest + updates, new weights file, new sha, re-signed or unsigned."""
    from safetensors.torch import save_file
    src_js = Path(src_js); man = json.loads(src_js.read_text(encoding="utf-8"))
    out_stem = Path(out_stem)
    if out_stem.suffix in (".json", ".safetensors"):
        out_stem = out_stem.with_suffix("")
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    st_p, js_p = Path(str(out_stem) + ".safetensors"), Path(str(out_stem) + ".json")
    if js_p.exists() or st_p.exists():
        raise SystemExit(f"{js_p} exists (bundles are never overwritten; pick a new --out)")
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(st_p), metadata={"format": S.ORGAN_FORMAT})
    tf = man.get("task_files") or {}
    for kk, info in list(tf.items()):   # carry shipped task files along under the new stem
        src = src_js.parent / info["file"]
        dst = Path(str(out_stem) + "." + info["file"].split(".")[-2] + ".jsonl")
        if src.exists():
            shutil.copy2(src, dst); tf[kk] = dict(file=dst.name, sha256=S.file_sha(dst, 64))
    man.update(man_updates)
    man.update(format=S.ORGAN_FORMAT, file=st_p.name, sha256=S.file_sha(st_p, 64), signature=None,
               derived_from=dict(file=src_js.name, sha256=json.loads(src_js.read_text(encoding="utf-8")).get("sha256")))
    if signer:
        man["signature"] = signer(S.bundle_sha_payload(man))
    js_p.write_text(json.dumps(man, indent=1), encoding="utf-8")
    return js_p


def cache_view(js, n=5):
    man, t, ver = verify_bundle(js)
    c = S.RehearsalCache.from_bundle(man, t)
    out = dict(format=man.get("format"), rehearsal=man.get("rehearsal"), verify=ver)
    if c is None:
        out["cache"] = None
        return out
    out["cache"] = c.stats()
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(S.HERE / "tokenizers" / "seed-bpe4096-v1" / "tokenizer.json"))
        order = torch.argsort(c.score, descending=True)
        ex = []
        for i in order[:n].tolist():
            ids = c.tokens[i].long().tolist(); m = c.mask[i].tolist()
            tgt = [ids[j + 1] for j in range(len(m)) if m[j]]
            ex.append(dict(score=round(float(c.score[i]), 3), source=S.CACHE_SRC_INV[int(c.source[i])],
                           text=tok.decode(ids[:64]).replace("\n", " ")[:200], targets=tok.decode(tgt[:24])[:120]))
        out["hardest"] = ex
    except Exception as e:
        out["hardest"] = f"(decode failed: {e})"
    return out


def cache_strip(js, out_stem, signer=None, reason="stripped: seller does not share training data"):
    man, t, ver = verify_bundle(js)
    t2 = {k: v for k, v in t.items() if not k.startswith("cache.")}
    tok = man.get("tokenizer") or {}
    return rewrite_bundle(js, out_stem, t2, dict(rehearsal=S.cache_manifest(None, tok, included=False, reason=reason)), signer)


def cache_rebuild(js, out_stem, host_path, socket_dir, task_path=None, dev="cuda", signer=None, cache_seqs=1024, cache_mb=4.0,
                  provenance=None, data_license=None):
    """Re-select the cache with the organ installed (full strength) in host_path, from a task file or synthetic samples."""
    spec, anchor, calib = S.load_socket(socket_dir)
    man, t, ver = verify_bundle(js)
    host, _ = load_host(host_path, dev)
    same = (man.get("donor") or {}).get("lineage") == host.lineage
    oid, rep = S.install_bundle(host, man, t, calib, align=False if same else "auto")
    host.to(dev); k = host.organ_ids.index(oid); set_alpha(host, k, 1.0)
    ctx = min(128, host.ctx); cap = S.cache_cap(cache_seqs, cache_mb, ctx)
    if task_path:
        cache = build_task_cache(host, k, Task(task_path, S.find_tokenizer(host)), ctx, dev, np.random.default_rng(0), cap,
                                 provenance=provenance or f"task file {Path(task_path).name}", data_license=data_license or "")
        mode = "full"
    else:
        cache = S.synth_cache(host, k, calib, dev, n=min(cap, 256), T=ctx, cap=cap, log=log); mode = "synthetic"
    cache.baseline = S.cache_loss(host, cache, dev, "check")
    t2 = {kk: v for kk, v in t.items() if not kk.startswith("cache.")}
    t2.update(cache.to_tensors())
    rh = S.cache_manifest(cache, host.tokenizer, mode=mode, note=f"rebuilt on host {host.lineage}")
    return rewrite_bundle(js, out_stem, t2, dict(rehearsal=rh), signer), cache.stats()


def eval_task(host_path, eval_path, data_dir, dev="cuda", ctx=128):
    host, info = load_host(host_path, dev)
    tok = S.find_tokenizer(host)
    core = Core(data_dir); cval = core.val(min(ctx, host.ctx), dev)
    return dict(host=str(host_path), shape=host.shape_str(), organs=host.organ_ids, step=info.get("step"),
                core=round(eval_core(host, cval), 4), task=Task(eval_path, tok).metric(host, dev))
