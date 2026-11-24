"""Chart a Seed run: loss curve + cell/layer count over time, growth events marked."""
import csv, json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

run = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/demo1")
rows = list(csv.DictReader(open(run / "metrics.csv")))
ev = [json.loads(l) for l in open(run / "events.jsonl")]
m = [float(r["minutes"]) for r in rows]
fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(m, [float(r["train_loss"]) for r in rows], color="#999", lw=1, label="train loss (EMA)")
ax.plot(m, [float(r["val_loss"]) for r in rows], color="#c0392b", lw=2, label="validation loss")
ax.set_xlabel("minutes on CPU"); ax.set_ylabel("loss (lower is better)")
ax.set_ylim(min(float(r["val_loss"]) for r in rows) - 0.2, min(8, max(float(r["val_loss"]) for r in rows) + 0.2))
ax2 = ax.twinx()
ax2.step(m, [int(r["cells"]) for r in rows], where="post", color="#2471a3", lw=2, label="cells")
ax2.step(m, [int(r["layers"]) for r in rows], where="post", color="#27ae60", lw=2, ls="--", label="layers")
ax2.set_ylabel("cells / layers"); ax2.set_ylim(0, max(int(r["cells"]) for r in rows) + 2)
colors = dict(duplicate="#2471a3", add_layer="#27ae60", prune="#8e44ad")
for e in ev:
    if e["kind"] in colors:
        ax.axvline(e["minutes"], color=colors[e["kind"]], alpha=0.35, lw=1)
        ax2.annotate({"duplicate": "split", "add_layer": "+layer", "prune": "prune"}[e["kind"]],
                     (e["minutes"], ax2.get_ylim()[1] - 0.6), rotation=90, fontsize=7, color=colors[e["kind"]], ha="right")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="center right")
last = rows[-1]
plt.title(f"Seed: one cell grows into {last['layers']} layers / {last['cells']} cells "
          f"({int(last['params'])/1e6:.2f}M params), CPU only")
plt.tight_layout(); out = run / "growth_chart.png"; plt.savefig(out, dpi=120); print(out)
