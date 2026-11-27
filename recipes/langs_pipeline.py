r"""Per-language seed pipeline v3 (2026-11-26 CST). Retrains seeds given on the command line (default c1 js1 ps1), one at a time, from scratch.
v2 bug fixed: v2 archived any run whose process had exited, so c1/js1/ps1 crashed at step 1500 (UnicodeEncodeError in a SAMPLE print) and were
marked done. v3 only archives a run when ALL hold: the stop came from a STOP file (plateau rule here, or the held-out watcher for ps1/sh1),
last metrics step >= 9500, and stderr.log has no Traceback. Otherwise: one automatic resume from latest.pt (if the crash looks transient),
then FAILED.txt (no final.pt, no ARCHIVED.txt). Launchers set PYTHONIOENCODING=utf-8 / PYTHONUTF8=1.
Plateau rule: min 9500, stop when val_mix_ema gains < 1.5% over 1500 steps (7 rows), cap 16000. After each good run: final.pt + lang_eval.py.
Log: D:\seed_v4\langs_pipeline.log. Completion marker: D:\seed_v4\LANGS_RETRAIN_DONE.txt (per-seed status + held-out bpb)."""
import os, sys, time, json, shutil, subprocess, csv
V = r"D:\seed_v4"; SEED = r"C:\Users\mcgar\Desktop\Seed"; PY = r"C:\Users\mcgar\AppData\Local\Programs\Python\Python312\python.exe"
LOG = open(os.path.join(V, "langs_pipeline.log"), "a", encoding="utf-8")
def log(s): LOG.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {s}\n"); LOG.flush()
def alive(pid): return str(pid) in subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout
def rows(d):
    p = os.path.join(d, "metrics.csv")
    try: return list(csv.DictReader(open(p))) if os.path.exists(p) else []
    except Exception: return []
def last_step(d): r = rows(d); return int(r[-1]["step"]) if r else 0
def flat(d, cap=16000):
    r = rows(d)
    if not r or int(r[-1]["step"]) < 9500: return False
    if int(r[-1]["step"]) >= cap: return True
    return len(r) >= 7 and (float(r[-7]["val_mix_ema"]) - float(r[-1]["val_mix_ema"])) / float(r[-7]["val_mix_ema"]) < 0.015
def crashed(d):
    p = os.path.join(d, "stderr.log"); return os.path.exists(p) and "Traceback" in open(p, encoding="utf-8", errors="ignore").read()
env = os.environ.copy(); env.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
seeds = sys.argv[1:] or ["c1", "js1", "ps1"]; status = {}
log(f"v3 start (retrain {seeds}); validity checks on; UTF-8 stdio")
for n in seeds:
    d = os.path.join(V, "seed_" + n)
    if os.path.exists(os.path.join(d, "ARCHIVED.txt")): log(f"{n} already archived (valid), skip"); status[n] = "already archived"; continue
    for attempt in (1, 2):
        if os.path.exists(os.path.join(d, "STOP")): shutil.move(os.path.join(d, "STOP"), os.path.join(d, f"STOP_old_{int(time.time())}"))
        if attempt == 2 and os.path.exists(os.path.join(d, "stderr.log")): shutil.move(os.path.join(d, "stderr.log"), os.path.join(d, f"stderr_attempt1_{int(time.time())}.log"))
        subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", os.path.join(SEED, f"run_seed_{n}.ps1")], env=env); time.sleep(5)
        pid = int(open(d + "_pid.txt").read().strip()); log(f"{n} launched pid {pid} (attempt {attempt}, from step {last_step(d)})"); t0 = time.time(); we_stopped = False
        while alive(pid):
            if flat(d):
                open(os.path.join(d, "STOP"), "w").write("plateau stop by langs pipeline v3\n"); we_stopped = True; log(f"{n} plateau STOP at step {last_step(d)}")
                for _ in range(360):
                    if not alive(pid): break
                    time.sleep(5)
                break
            if not rows(d) and time.time() - t0 > 1800: log(f"{n} no metrics after 30 min"); break
            time.sleep(60)
        time.sleep(5); st = last_step(d); stop_file = os.path.exists(os.path.join(d, "STOP")); bad = crashed(d)
        ok = stop_file and st >= 9500 and not bad and os.path.exists(os.path.join(d, "latest.pt"))
        log(f"{n} exited: step {st}, STOP file {stop_file} (by pipeline {we_stopped}), traceback {bad} -> {'VALID' if ok else 'NOT valid'}")
        if ok: break
        if alive(pid): log(f"{n} still alive without valid stop; leaving it and marking FAILED"); break
        if attempt == 1 and os.path.exists(os.path.join(d, "latest.pt")): log(f"{n} retrying once (resume from latest.pt)"); continue
    if ok:
        r = rows(d); shutil.copy2(os.path.join(d, "latest.pt"), os.path.join(d, "final.pt"))
        open(os.path.join(d, "ARCHIVED.txt"), "w").write(f"Archived {time.strftime('%Y-%m-%d %H:%M')} CST by langs pipeline v3 (validated: STOP, step>=9500, no traceback).\nfinal metrics row: {json.dumps(r[-1])}\n")
        log(f"{n} archived step {r[-1]['step']} shape {r[-1].get('layers')}x{r[-1].get('organs')} val_mix_ema {r[-1]['val_mix_ema']} step_ms {r[-1]['step_ms']}")
        rc = subprocess.run([PY, "-W", "ignore", os.path.join(V, "lang_eval.py"), "seed_" + n], cwd=SEED, capture_output=True, text=True, env=env)
        log(f"{n} held-out eval rc {rc.returncode}: {(rc.stdout or rc.stderr).strip()[-300:]}"); status[n] = dict(state="done", step=r[-1]["step"], eval=(rc.stdout or rc.stderr).strip()[-300:])
    else:
        open(os.path.join(d, "FAILED.txt"), "w").write(f"{time.strftime('%Y-%m-%d %H:%M')} step {last_step(d)}; see stderr.log\n"); log(f"{n} FAILED - see {d}\\stderr.log"); status[n] = dict(state="FAILED", step=last_step(d))
open(os.path.join(V, "LANGS_RETRAIN_DONE.txt"), "w").write(time.strftime("%Y-%m-%d %H:%M") + "\n" + json.dumps(status, indent=1) + "\n"); log("LANGS RETRAIN DONE " + json.dumps(status)[:400])
