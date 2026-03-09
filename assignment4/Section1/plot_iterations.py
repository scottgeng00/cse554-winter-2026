import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser(description="Plot iteration times from CSV files")
p.add_argument("csvs", nargs="+", help="CSV files from profiler_code.py --csv")
p.add_argument("-o", "--output", default="iteration_times.png")
args = p.parse_args()

fig, ax = plt.subplots(figsize=(10, 5))

for path in args.csvs:
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    methods = dict.fromkeys(r["method"] for r in rows)
    for method in methods:
        iters = [(int(r["iteration"]), float(r["time_ms"])) for r in rows if r["method"] == method]
        iters.sort()
        xs, ys = zip(*iters)
        ax.scatter(xs, ys, s=4, alpha=0.6, label=method)
        print(f"{method}: {len(iters)} iters, mean={np.mean(ys):.2f}ms, max={np.max(ys):.2f}ms")

ax.set_xlabel("Iteration ID")
ax.set_yscale("log")
ax.set_ylabel("Iteration time (ms)")
ax.set_title("Per-iteration time comparison")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(args.output, dpi=150)
print(f"Saved {args.output}")
