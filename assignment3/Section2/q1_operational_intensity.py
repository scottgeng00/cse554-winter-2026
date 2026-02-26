#!/usr/bin/env python3
"""Q1: Operational Intensity of Prefill and Decode Attention."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = {
    "LLaMA3-1B": {"num_qo_heads": 32, "num_kv_heads": 8, "head_dim": 64},
    "LLaMA3-3B": {"num_qo_heads": 24, "num_kv_heads": 8, "head_dim": 128},
    "LLaMA3-8B": {"num_qo_heads": 32, "num_kv_heads": 8, "head_dim": 128},
}


def prefill_oi(num_qo_heads, num_kv_heads, head_dim, p):
    """OI = p * H_q / (H_q + H_kv)."""
    flops = 4.0 * p * p * head_dim * num_qo_heads
    bytes_transferred = 4.0 * p * head_dim * (num_qo_heads + num_kv_heads)
    return flops / bytes_transferred


def decode_oi(num_qo_heads, num_kv_heads, head_dim, c):
    """OI = H_q * c / (H_q + H_kv * c)."""
    flops = 4.0 * num_qo_heads * c * head_dim
    bytes_transferred = 2.0 * head_dim * (2 * num_qo_heads + 2 * num_kv_heads * c)
    return flops / bytes_transferred


seq_lens = (2 ** np.arange(7, 16)).astype(int)

print("=" * 80)
print("PREFILL ATTENTION — Operational Intensity (FLOPs / Byte)")
print("  OI = p * H_q / (H_q + H_kv)")
print("=" * 80)
header = f"{'p':>8}" + "".join(f"  {name:>14}" for name in MODELS)
print(header)
print("-" * len(header))
prefill_results = {name: [] for name in MODELS}
for p in seq_lens:
    row = f"{p:>8}"
    for name, cfg in MODELS.items():
        oi = prefill_oi(**cfg, p=int(p))
        prefill_results[name].append(oi)
        row += f"  {oi:>14.2f}"
    print(row)

print()
print("=" * 80)
print("DECODE ATTENTION — Operational Intensity (FLOPs / Byte)")
print("  OI = H_q * c / (H_q + H_kv * c),  asymptote → H_q / H_kv")
print("=" * 80)
header = f"{'c':>8}" + "".join(f"  {name:>14}" for name in MODELS)
print(header)
print("-" * len(header))
decode_results = {name: [] for name in MODELS}
for c in seq_lens:
    row = f"{c:>8}"
    for name, cfg in MODELS.items():
        oi = decode_oi(**cfg, c=int(c))
        decode_results[name].append(oi)
        row += f"  {oi:>14.4f}"
    print(row)

print()
print("Asymptotic decode OI (c → ∞):")
for name, cfg in MODELS.items():
    print(f"  {name}: H_q/H_kv = {cfg['num_qo_heads']}/{cfg['num_kv_heads']}"
          f" = {cfg['num_qo_heads'] / cfg['num_kv_heads']:.1f}")

STYLES = {
    "LLaMA3-1B": dict(marker="o", linestyle="-",  linewidth=2.5, markersize=8, color="tab:blue"),
    "LLaMA3-3B": dict(marker="s", linestyle="--", linewidth=2.0, markersize=7, color="tab:orange"),
    "LLaMA3-8B": dict(marker="^", linestyle=":",  linewidth=2.0, markersize=7, color="tab:green"),
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for name in MODELS:
    ax.plot(np.log2(seq_lens), prefill_results[name], label=name, **STYLES[name])
ax.set_xlabel(r"$\log_2(p)$")
ax.set_ylabel("Operational Intensity (FLOPs / Byte)")
ax.set_title("Prefill Attention OI")
ax.legend()
ax.grid(True)
ax.set_xticks(np.arange(7, 16))

ax = axes[1]
for name in MODELS:
    ax.plot(np.log2(seq_lens), decode_results[name], label=name, **STYLES[name])
ax.set_xlabel(r"$\log_2(c)$")
ax.set_ylabel("Operational Intensity (FLOPs / Byte)")
ax.set_title("Decode Attention OI")
ax.legend()
ax.grid(True)
ax.set_xticks(np.arange(7, 16))

fig.suptitle("Operational Intensity of Attention (Q1)", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("q1_operational_intensity.png", dpi=200)
print("\nSaved plot → q1_operational_intensity.png")
