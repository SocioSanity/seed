r"""Seed parts tool: inspect, validate, export and score hot-swappable parts.

  python parts_tool.py inspect  PART.json
  python parts_tool.py validate PART.json --base latest.pt|tree.json
  python parts_tool.py export   --from latest.pt --kind organ|cell|tree [--id o3|c17] --out parts/exports [--raw]
  python parts_tool.py score    PART.json --base tree.json [--data DIR] [--scores-dir parts/scores]
  python parts_tool.py protect  PART.json [--off]          (sets the protected tag in the manifest)

Portable organs (seed-socket-v1, see SOCKET_SPEC.md / ORGANS_README.md):
  python parts_tool.py make-socket  --from runs/run1/latest.pt            (once: publish anchor + calibration set)
  python parts_tool.py keygen       --author NAME                                 (ed25519 key in organs/keys)
  python parts_tool.py train-organ  --host HOST.pt --task T.jsonl --task-eval E.jsonl --name NAME --out STEM [--steps 1500]
  python parts_tool.py harvest      --host HOST.pt --out STEM [--organ o3 | --task T.jsonl] [--name ...]
  python parts_tool.py verify       BUNDLE.json
  python parts_tool.py inspect      BUNDLE.json
  python parts_tool.py import       BUNDLE.json|NAME[@VER] --host HOST.pt --out NEW_HOST.pt [--steps 300] [--no-align]
  python parts_tool.py publish      BUNDLE.json                                   (copy into organs/library + index.json)
  python parts_tool.py list
  python parts_tool.py cache        BUNDLE.json [--action view|strip|rebuild] [--out NEW_STEM] [--host HOST.pt --task T.jsonl]
  python parts_tool.py eval-task    --host HOST.pt --task-eval E.jsonl              (task metric + core loss of any host)
Rehearsal cache (seed-organ/1.1): train-organ/harvest add --cache-mode full|synthetic|none, --cache-seqs, --cache-mb,
  --data-provenance, --data-license; train-organ --no-task-files; import --report-eval E.jsonl --ignore-task-files
Score = how much the part lowers held-out TinyStories loss when plugged (at full strength, no training)
into a base Seed (base loss vs loss with the part). Written as JSON to --scores-dir.
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
from seed_v4 import (load_model_any, install_parts_full, read_part, validate_part, export_cell, export_organ, export_tree,
                     evaluate, nullctx, file_sha, PART_FORMAT, HERE)


def val_batches(data_dir, ctx, n=12, B=16, seed=999, dev="cpu"):
    a = np.memmap(Path(data_dir) / "valid.bin", dtype=np.uint16, mode="r")
    r = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        s = r.integers(0, len(a) - ctx - 1, size=B)
        x = torch.tensor(np.stack([a[i:i + ctx + 1] for i in s]).astype(np.int64))
        out.append((x[:, :-1].to(dev), x[:, 1:].to(dev)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["inspect", "validate", "export", "score", "protect", "make-socket", "keygen", "train-organ", "cache", "eval-task",
                                    "harvest", "verify", "import", "publish", "list"])
    ap.add_argument("part", nargs="?")
    ap.add_argument("--base"); ap.add_argument("--from", dest="src"); ap.add_argument("--kind", default="organ")
    ap.add_argument("--id", default=None); ap.add_argument("--out", default=str(HERE / "parts" / "exports"))
    ap.add_argument("--raw", action="store_true"); ap.add_argument("--off", action="store_true")
    ap.add_argument("--data", default=str(HERE / "data")); ap.add_argument("--ctx", type=int, default=0)
    ap.add_argument("--scores-dir", default=str(HERE / "parts" / "scores")); ap.add_argument("--device", default="auto")
    ap.add_argument("--host"); ap.add_argument("--task"); ap.add_argument("--task-eval"); ap.add_argument("--organ")
    ap.add_argument("--name"); ap.add_argument("--version", default="0.1.0"); ap.add_argument("--description", default="")
    ap.add_argument("--author", default="unknown"); ap.add_argument("--license", default="CC-BY-4.0")
    ap.add_argument("--price", default=None, help="marketplace placeholder (no payment code)")
    ap.add_argument("--sign-key", default=None, help="ed25519 key file from keygen")
    ap.add_argument("--steps", type=int, default=0); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--tol", type=float, default=0.02); ap.add_argument("--min-gain", type=float, default=0.02)
    ap.add_argument("--no-align", action="store_true"); ap.add_argument("--force-align", action="store_true"); ap.add_argument("--control", default=None, choices=[None, "random"])
    ap.add_argument("--socket-dir", default=str(HERE / "socket" / "seed-socket-v1"))
    ap.add_argument("--library", default=str(HERE / "organs" / "library")); ap.add_argument("--rejected-dir", default=str(HERE / "organs" / "rejected"))
    ap.add_argument("--cache-mode", default="full", choices=["full", "synthetic", "none"], help="rehearsal cache shipped in the bundle")
    ap.add_argument("--cache-seqs", type=int, default=1024); ap.add_argument("--cache-mb", type=float, default=4.0)
    ap.add_argument("--data-provenance", default=None, help="seller's declaration of where the cached data comes from")
    ap.add_argument("--data-license", default=None); ap.add_argument("--no-task-files", action="store_true", help="train-organ: ship no tune/eval JSONL")
    ap.add_argument("--report-eval", default=None, help="import: extra eval JSONL for reporting only")
    ap.add_argument("--ignore-task-files", action="store_true", help="import: do not use the bundle's shipped tune/eval files")
    ap.add_argument("--action", default="view", choices=["view", "strip", "rebuild"], help="cache subcommand action"); ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--result-json", default=None); ap.add_argument("--keys-dir", default=str(HERE / "organs" / "keys"))
    a = ap.parse_args()
    if a.device == "auto":
        a.device = "cuda" if torch.cuda.is_available() else "cpu"
    if a.cmd in ("make-socket", "keygen", "train-organ", "harvest", "verify", "import", "publish", "list", "cache", "eval-task") or (
            a.cmd == "inspect" and a.part and str(json.loads(Path(a.part).read_text(encoding="utf-8")).get("format", "")).startswith("seed-organ/")):
        return organ_cmd(a)
    torch.set_num_threads(2)
    if a.cmd == "inspect":
        man, t = read_part(a.part)
        show = {k: v for k, v in man.items() if k != "structure"}
        print(json.dumps(show, indent=1))
        print(f"{len(t)} tensors, {sum(v.numel() for v in t.values()):,} values; sha ok")
    elif a.cmd == "validate":
        man, t = read_part(a.part)
        m, _ = load_model_any(a.base)
        errs, warns = validate_part(m, man, t)
        print("OK: compatible" if not errs else "REFUSED:\n  " + "\n  ".join(errs))
        for w in warns:
            print("warning:", w)
        sys.exit(1 if errs else 0)
    elif a.cmd == "protect":
        p = Path(a.part); man = json.loads(p.read_text(encoding="utf-8")); man["protected"] = not a.off
        p.write_text(json.dumps(man, indent=1), encoding="utf-8"); print(f"protected={man['protected']}")
    elif a.cmd == "export":
        m, info = load_model_any(a.src, use_ema=not a.raw)
        tag = f"step{info['step']}_{m.lineage}"
        out = Path(a.out)
        extra = dict(step=info["step"], ema=info["ema"], exported_by="parts_tool")
        if a.kind == "tree":
            p = export_tree(m, out / f"tree_{m.shape_str()}_{tag}", extra)
        elif a.kind == "organ":
            k = m.organ_ids.index(a.id) if a.id else len(m.organ_ids) - 1
            p = export_organ(m, k, out / f"organ_{m.organ_ids[k]}_{tag}", extra)
        else:
            li, ci = m.find_cell(a.id) if a.id else (0, 0)
            p = export_cell(m, li, ci, out / f"cell_{m.layers[li].cells[ci].cid}_{tag}", extra)
        print(p)
    elif a.cmd == "score":
        man, _ = read_part(a.part)
        base, binfo = load_model_any(a.base, device=a.device)
        vb = val_batches(a.data, a.ctx or min(base.ctx, 256), dev=a.device)
        l0 = evaluate(base, vb, nullctx)
        m, _ = load_model_any(a.base, device=a.device)
        added = install_parts_full(m, [a.part])
        l1 = evaluate(m, vb, nullctx)
        res = dict(part=str(a.part), part_id=man["id"], part_kind=man["kind"], part_sha256=man["sha256"], base=str(a.base),
                   base_shape=base.shape_str(), base_lineage=base.lineage, same_lineage=man.get("lineage") == base.lineage,
                   val_ts_base=round(l0, 4), val_ts_with_part=round(l1, 4), gain=round(l0 - l1, 4),
                   rel_gain_pct=round(100 * (l0 - l1) / l0, 2), installed=added, scored=time.strftime("%Y-%m-%d %H:%M:%S"),
                   note="zero-shot: part plugged in at full strength, no training. Negative gain = it hurts this base as-is.")
        sd = Path(a.scores_dir); sd.mkdir(parents=True, exist_ok=True)
        outp = sd / f"score_{man['id']}_{man['sha256'][:8]}_on_{base.lineage}.json"
        outp.write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(json.dumps(res, indent=1)); print(outp)


def organ_cmd(a):
    import seed_organs as O
    dev = "cuda" if (a.device == "cuda" and torch.cuda.is_available()) else "cpu"
    signer = O.make_signer(a.sign_key, a.author) if a.sign_key else None
    meta = lambda: {k: v for k, v in dict(name=a.name, version=a.version, description=a.description, author=a.author,
                                           license=a.license, price=a.price).items() if v is not None}
    res = None
    if a.cmd == "make-socket":
        from tokenizers import Tokenizer
        m, info = load_model_any(a.src, use_ema=True)
        tdir = HERE / "tokenizers" / "seed-bpe4096-v1"
        tman = json.loads((tdir / "seed_tokenizer_manifest.json").read_text())
        valid = np.memmap(Path(a.data) / "valid.bin", dtype=np.uint16, mode="r")
        res = O.S.create_socket(a.socket_dir, m.emb.weight, valid, tman, source=f"{a.src} (EMA, step {info['step']})",
                                readout_gain=m.nf)
        res = {k: v for k, v in res.items() if not k.startswith("_")}
    elif a.cmd == "keygen":
        kp, pub = O.keygen(a.author, a.keys_dir); res = dict(key=str(kp), public_key=pub, note="keep the .key file private")
    elif a.cmd == "train-organ":
        js, r = O.train_organ(a.host, a.task, a.task_eval, a.out, a.data, a.socket_dir, steps=a.steps or 1500, batch=a.batch,
                              ctx=a.ctx or 128, meta=meta(), dev=dev, signer=signer, cache_mode=a.cache_mode, cache_seqs=a.cache_seqs,
                              cache_mb=a.cache_mb, provenance=a.data_provenance, data_license=a.data_license, ship_files=not a.no_task_files)
        res = dict(bundle=str(js), **r)
    elif a.cmd == "harvest":
        js = O.harvest(a.host, a.out, a.socket_dir, organ=a.organ, task_path=a.task, data_dir=a.data, meta=meta(), dev=dev, signer=signer,
                       cache_mode=a.cache_mode, cache_seqs=a.cache_seqs, cache_mb=a.cache_mb)
        if a.task and a.task_eval:
            O.ship_task_files(js, a.task, a.task_eval)
            if signer:
                O.sign_bundle(js, signer)
        res = dict(bundle=str(js))
    elif a.cmd in ("verify", "inspect"):
        try:
            man, t, ver = O.verify_bundle(a.part)
        except ValueError as e:
            print(json.dumps(dict(ok=False, error=str(e)))); sys.exit(3)
        cc = O.S.RehearsalCache.from_bundle(man, t)
        res = dict(verify=ver, manifest={k: v for k, v in man.items() if k != "signature"},
                   tensors=len(t), calib_layers=sum(1 for k in t if k.startswith("calib.")),
                   cache=None if cc is None else cc.stats())
    elif a.cmd == "import":
        js = O.resolve(a.part, a.library)
        res = O.adopt(a.host, js, a.out, a.data, a.socket_dir, steps=a.steps or 300, batch=a.batch, ctx=a.ctx or 128, tol=a.tol,
                      min_gain=a.min_gain, align=False if a.no_align else ("auto" if not a.force_align else True), dev=dev, control=a.control,
                      rej_dir=a.rejected_dir, report_eval=a.report_eval, use_task_files=not a.ignore_task_files)
    elif a.cmd == "cache":
        if a.action == "view":
            res = O.cache_view(a.part, a.n)
        elif a.action == "strip":
            res = dict(bundle=str(O.cache_strip(a.part, a.out, signer)))
        else:
            js, st = O.cache_rebuild(a.part, a.out, a.host, a.socket_dir, task_path=a.task, dev=dev, signer=signer, cache_seqs=a.cache_seqs,
                                     cache_mb=a.cache_mb, provenance=a.data_provenance, data_license=a.data_license)
            res = dict(bundle=str(js), cache=st)
    elif a.cmd == "eval-task":
        res = O.eval_task(a.host, a.task_eval, a.data, dev=dev, ctx=a.ctx or 128)
    elif a.cmd == "publish":
        res = dict(published=str(O.publish(a.part, a.library)))
    elif a.cmd == "list":
        idx = O.lib_index(a.library)
        for o in idx["organs"]:
            print(f"{o['name']}@{o['version']}  {o['n_params']/1e6:.2f}M  {o.get('license')}  price={o.get('price')}  "
                  f"signed={o['signed']}  task={o.get('task_after')}  - {o.get('description')}")
        res = dict(count=len(idx["organs"]))
    print(json.dumps(res, indent=1, default=str))
    if a.result_json:
        Path(a.result_json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.result_json).write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    if a.cmd == "import" and not res.get("accepted"):
        sys.exit(2)


if __name__ == "__main__":
    main()
