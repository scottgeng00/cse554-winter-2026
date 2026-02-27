#!/usr/bin/env python3
"""Q2: Benchmark Prefill Attention — torch SDPA vs FlashInfer."""

import torch
import torch.nn.functional as F
import numpy as np
import json
import flashinfer

MODELS = {
    "LLaMA3-1B": {"num_qo_heads": 32, "num_kv_heads": 8, "head_dim": 64},
    "LLaMA3-3B": {"num_qo_heads": 24, "num_kv_heads": 8, "head_dim": 128},
    "LLaMA3-8B": {"num_qo_heads": 32, "num_kv_heads": 8, "head_dim": 128},
}

WARMUP = 20
REPEAT = 100
DTYPE = torch.float16
DEVICE = "cuda"

def _jit_warmup_flashinfer():
    print("Pre-warming FlashInfer JIT kernels …")
    for p in (2 ** np.arange(7, 16)).astype(int):
        for cfg in MODELS.values():
            Hq, Hkv, d = cfg["num_qo_heads"], cfg["num_kv_heads"], cfg["head_dim"]
            q = torch.randn(p, Hq, d, dtype=DTYPE, device=DEVICE)
            k = torch.randn(p, Hkv, d, dtype=DTYPE, device=DEVICE)
            v = torch.randn(p, Hkv, d, dtype=DTYPE, device=DEVICE)
            flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True)
    torch.cuda.synchronize()
    print("JIT warmup done.")


def prefill_flops(batch_size, num_qo_heads, head_dim, p):
    return 4 * batch_size * num_qo_heads * head_dim * p * p


def sdpa_prefill(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)


def bench_sdpa_prefill(batch_size, num_qo_heads, num_kv_heads, head_dim, p):
    q = torch.randn(batch_size, num_qo_heads, p, head_dim, dtype=DTYPE, device=DEVICE)
    k = torch.randn(batch_size, num_kv_heads, p, head_dim, dtype=DTYPE, device=DEVICE)
    v = torch.randn(batch_size, num_kv_heads, p, head_dim, dtype=DTYPE, device=DEVICE)

    for _ in range(WARMUP):
        sdpa_prefill(q, k, v)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(REPEAT):
        sdpa_prefill(q, k, v)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / REPEAT


def bench_fi_prefill_single(num_qo_heads, num_kv_heads, head_dim, p):
    q = torch.randn(p, num_qo_heads, head_dim, dtype=DTYPE, device=DEVICE)
    k = torch.randn(p, num_kv_heads, head_dim, dtype=DTYPE, device=DEVICE)
    v = torch.randn(p, num_kv_heads, head_dim, dtype=DTYPE, device=DEVICE)

    for _ in range(WARMUP):
        flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(REPEAT):
        flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / REPEAT


def bench_fi_prefill_batch(batch_size, num_qo_heads, num_kv_heads, head_dim, p):
    total = batch_size * p
    q = torch.randn(total, num_qo_heads, head_dim, dtype=DTYPE, device=DEVICE)
    k = torch.randn(total, num_kv_heads, head_dim, dtype=DTYPE, device=DEVICE)
    v = torch.randn(total, num_kv_heads, head_dim, dtype=DTYPE, device=DEVICE)

    qo_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device=DEVICE) * p
    kv_indptr = qo_indptr.clone()

    workspace = torch.empty(128 << 20, dtype=torch.uint8, device=DEVICE)
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace, "NHD")
    wrapper.plan(
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        causal=True,
    )

    for _ in range(WARMUP):
        wrapper.run(q, k, v)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(REPEAT):
        wrapper.run(q, k, v)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / REPEAT


def run_exp1():
    print("\n" + "=" * 70)
    print("Experiment 1: bs=1, varying prompt length p")
    print("=" * 70)
    p_values = (2 ** np.arange(7, 16)).astype(int)
    results = {}

    for name, cfg in MODELS.items():
        H_q, H_kv, d = cfg["num_qo_heads"], cfg["num_kv_heads"], cfg["head_dim"]
        sdpa_tflops, fi_tflops = [], []
        for p in p_values:
            flops = prefill_flops(1, H_q, d, int(p))

            ms_sdpa = bench_sdpa_prefill(1, H_q, H_kv, d, int(p))
            tf_sdpa = flops / (ms_sdpa * 1e-3) / 1e12
            sdpa_tflops.append(tf_sdpa)

            ms_fi = bench_fi_prefill_single(H_q, H_kv, d, int(p))
            tf_fi = flops / (ms_fi * 1e-3) / 1e12
            fi_tflops.append(tf_fi)

            print(f"  {name}  p={p:>5}  SDPA={tf_sdpa:7.2f} TFlops  "
                  f"FlashInfer={tf_fi:7.2f} TFlops")

        results[name] = {"sdpa": sdpa_tflops, "flashinfer": fi_tflops}

    return p_values, results


def run_exp2():
    print("\n" + "=" * 70)
    print("Experiment 2: varying batch size, p=1024")
    print("=" * 70)
    bs_values = (2 ** np.arange(0, 7)).astype(int)
    p = 1024
    results = {}

    for name, cfg in MODELS.items():
        H_q, H_kv, d = cfg["num_qo_heads"], cfg["num_kv_heads"], cfg["head_dim"]
        sdpa_tflops, fi_tflops = [], []
        for bs in bs_values:
            flops = prefill_flops(int(bs), H_q, d, p)

            ms_sdpa = bench_sdpa_prefill(int(bs), H_q, H_kv, d, p)
            tf_sdpa = flops / (ms_sdpa * 1e-3) / 1e12
            sdpa_tflops.append(tf_sdpa)

            ms_fi = bench_fi_prefill_batch(int(bs), H_q, H_kv, d, p)
            tf_fi = flops / (ms_fi * 1e-3) / 1e12
            fi_tflops.append(tf_fi)

            print(f"  {name}  bs={bs:>3}  SDPA={tf_sdpa:7.2f} TFlops  "
                  f"FlashInfer={tf_fi:7.2f} TFlops")

        results[name] = {"sdpa": sdpa_tflops, "flashinfer": fi_tflops}

    return bs_values, results


if __name__ == "__main__":
    _jit_warmup_flashinfer()

    p_values, res1 = run_exp1()
    bs_values, res2 = run_exp2()

    out = {
        "exp1_vary_p": {
            "p_values": p_values.tolist(),
            "results": {k: {kk: [float(v) for v in vv] for kk, vv in v.items()}
                        for k, v in res1.items()},
        },
        "exp2_vary_bs": {
            "bs_values": bs_values.tolist(),
            "results": {k: {kk: [float(v) for v in vv] for kk, vv in v.items()}
                        for k, v in res2.items()},
        },
    }
    with open("q2_prefill_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved → q2_prefill_results.json")
