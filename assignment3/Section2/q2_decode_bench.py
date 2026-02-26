#!/usr/bin/env python3
"""Q2: Benchmark Decode Attention — torch SDPA vs FlashInfer."""

import torch
import torch.nn.functional as F
import numpy as np
import json
import math
import flashinfer

MODELS = {
    "LLaMA3-1B": {"num_qo_heads": 32, "num_kv_heads": 8, "head_dim": 64},
    "LLaMA3-3B": {"num_qo_heads": 24, "num_kv_heads": 8, "head_dim": 128},
    "LLaMA3-8B": {"num_qo_heads": 32, "num_kv_heads": 8, "head_dim": 128},
}

WARMUP = 20
REPEAT = 200
DTYPE = torch.float16
DEVICE = "cuda"

_sdpa_supports_gqa = False
try:
    _q = torch.randn(1, 2, 1, 4, device=DEVICE, dtype=DTYPE)
    _k = torch.randn(1, 1, 1, 4, device=DEVICE, dtype=DTYPE)
    F.scaled_dot_product_attention(_q, _k, _k, enable_gqa=True)
    _sdpa_supports_gqa = True
except TypeError:
    pass
del _q, _k
print(f"torch SDPA enable_gqa support: {_sdpa_supports_gqa}")


def _jit_warmup_flashinfer():
    print("Pre-warming FlashInfer JIT kernels …")
    for c in (2 ** np.arange(7, 16)).astype(int):
        for cfg in MODELS.values():
            Hq, Hkv, d = cfg["num_qo_heads"], cfg["num_kv_heads"], cfg["head_dim"]
            q = torch.randn(Hq, d, dtype=DTYPE, device=DEVICE)
            k = torch.randn(c, Hkv, d, dtype=DTYPE, device=DEVICE)
            v = torch.randn(c, Hkv, d, dtype=DTYPE, device=DEVICE)
            flashinfer.single_decode_with_kv_cache(q, k, v)
    torch.cuda.synchronize()
    print("JIT warmup done.")


def decode_bytes(batch_size, num_qo_heads, num_kv_heads, head_dim, c):
    return batch_size * 2 * head_dim * (2 * num_qo_heads + 2 * num_kv_heads * c)


def sdpa_decode(q, k, v, num_qo_heads, num_kv_heads):
    if _sdpa_supports_gqa:
        return F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
    else:
        g = num_qo_heads // num_kv_heads
        return F.scaled_dot_product_attention(
            q, k.repeat_interleave(g, dim=1), v.repeat_interleave(g, dim=1))


def bench_sdpa_decode(batch_size, num_qo_heads, num_kv_heads, head_dim, c):
    q = torch.randn(batch_size, num_qo_heads, 1, head_dim, dtype=DTYPE, device=DEVICE)
    k = torch.randn(batch_size, num_kv_heads, c, head_dim, dtype=DTYPE, device=DEVICE)
    v = torch.randn(batch_size, num_kv_heads, c, head_dim, dtype=DTYPE, device=DEVICE)

    for _ in range(WARMUP):
        sdpa_decode(q, k, v, num_qo_heads, num_kv_heads)
    torch.cuda.synchronize()

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(REPEAT):
        sdpa_decode(q, k, v, num_qo_heads, num_kv_heads)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / REPEAT


def bench_fi_decode_single(num_qo_heads, num_kv_heads, head_dim, c):
    q = torch.randn(num_qo_heads, head_dim, dtype=DTYPE, device=DEVICE)
    k = torch.randn(c, num_kv_heads, head_dim, dtype=DTYPE, device=DEVICE)
    v = torch.randn(c, num_kv_heads, head_dim, dtype=DTYPE, device=DEVICE)

    for _ in range(WARMUP):
        flashinfer.single_decode_with_kv_cache(q, k, v)
    torch.cuda.synchronize()

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(REPEAT):
        flashinfer.single_decode_with_kv_cache(q, k, v)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / REPEAT


def bench_fi_decode_batch(batch_size, num_qo_heads, num_kv_heads, head_dim, c,
                          page_size=16):
    pages_per_req = math.ceil(c / page_size)
    total_pages = batch_size * pages_per_req
    last_pl = c % page_size
    if last_pl == 0:
        last_pl = page_size

    k_data = torch.randn(total_pages, num_kv_heads, page_size, head_dim,
                          dtype=DTYPE, device=DEVICE)
    v_data = torch.randn_like(k_data)

    kv_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device=DEVICE) * pages_per_req
    kv_indices = torch.arange(total_pages, dtype=torch.int32, device=DEVICE)
    kv_last_page_len = torch.full((batch_size,), last_pl, dtype=torch.int32, device=DEVICE)

    q = torch.randn(batch_size, num_qo_heads, head_dim, dtype=DTYPE, device=DEVICE)

    workspace = torch.empty(128 << 20, dtype=torch.uint8, device=DEVICE)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, "HND", use_tensor_cores=True)
    wrapper.plan(
        indptr=kv_indptr,
        indices=kv_indices,
        last_page_len=kv_last_page_len,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        data_type=DTYPE,
    )

    for _ in range(WARMUP):
        wrapper.run(q, (k_data, v_data))
    torch.cuda.synchronize()

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(REPEAT):
        wrapper.run(q, (k_data, v_data))
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / REPEAT


def run_exp1():
    print("\n" + "=" * 70)
    print("Experiment 1 (decode): bs=1, varying context length c")
    print("=" * 70)
    c_values = (2 ** np.arange(7, 16)).astype(int)
    results = {}

    for name, cfg in MODELS.items():
        H_q, H_kv, d = cfg["num_qo_heads"], cfg["num_kv_heads"], cfg["head_dim"]
        sdpa_gbps, fi_gbps = [], []
        for c in c_values:
            nbytes = decode_bytes(1, H_q, H_kv, d, int(c))

            ms = bench_sdpa_decode(1, H_q, H_kv, d, int(c))
            bw = nbytes / (ms * 1e-3) / 1e9
            sdpa_gbps.append(bw)

            ms = bench_fi_decode_single(H_q, H_kv, d, int(c))
            bw_fi = nbytes / (ms * 1e-3) / 1e9
            fi_gbps.append(bw_fi)

            print(f"  {name}  c={c:>5}  SDPA={bw:8.1f} GB/s  FlashInfer={bw_fi:8.1f} GB/s")

        results[name] = {"sdpa": sdpa_gbps, "flashinfer": fi_gbps}
    return c_values, results


def run_exp2():
    print("\n" + "=" * 70)
    print("Experiment 2 (decode): varying batch size, c=1024")
    print("=" * 70)
    bs_values = (2 ** np.arange(0, 7)).astype(int)
    c = 1024
    results = {}

    for name, cfg in MODELS.items():
        H_q, H_kv, d = cfg["num_qo_heads"], cfg["num_kv_heads"], cfg["head_dim"]
        sdpa_gbps, fi_gbps = [], []
        for bs in bs_values:
            nbytes = decode_bytes(int(bs), H_q, H_kv, d, c)

            ms = bench_sdpa_decode(int(bs), H_q, H_kv, d, c)
            bw = nbytes / (ms * 1e-3) / 1e9
            sdpa_gbps.append(bw)

            ms = bench_fi_decode_batch(int(bs), H_q, H_kv, d, c, page_size=16)
            bw_fi = nbytes / (ms * 1e-3) / 1e9
            fi_gbps.append(bw_fi)

            print(f"  {name}  bs={bs:>3}  SDPA={bw:8.1f} GB/s  FlashInfer={bw_fi:8.1f} GB/s")

        results[name] = {"sdpa": sdpa_gbps, "flashinfer": fi_gbps}
    return bs_values, results


def run_exp3():
    print("\n" + "=" * 70)
    print("Experiment 3 (decode): bs=128, c=1024, varying page_size (FlashInfer)")
    print("=" * 70)
    page_sizes = [1, 2, 4, 8, 16]
    bs, c = 128, 1024
    results = {}

    for name, cfg in MODELS.items():
        H_q, H_kv, d = cfg["num_qo_heads"], cfg["num_kv_heads"], cfg["head_dim"]
        fi_gbps = []
        nbytes = decode_bytes(bs, H_q, H_kv, d, c)
        for ps in page_sizes:
            ms = bench_fi_decode_batch(bs, H_q, H_kv, d, c, page_size=ps)
            bw = nbytes / (ms * 1e-3) / 1e9
            fi_gbps.append(bw)
            print(f"  {name}  page_size={ps:>2}  FlashInfer={bw:8.1f} GB/s")
        results[name] = {"flashinfer": fi_gbps}
    return page_sizes, results


if __name__ == "__main__":
    _jit_warmup_flashinfer()

    c_vals, res1 = run_exp1()
    bs_vals, res2 = run_exp2()
    ps_vals, res3 = run_exp3()

    def to_list(d):
        if isinstance(d, dict):
            return {k: to_list(v) for k, v in d.items()}
        if isinstance(d, (list, np.ndarray)):
            return [float(x) for x in d]
        return d

    out = {
        "exp1_vary_c":         {"c_values": list(map(int, c_vals)),  "results": to_list(res1)},
        "exp2_vary_bs":        {"bs_values": list(map(int, bs_vals)), "results": to_list(res2)},
        "exp3_vary_pagesize":  {"page_sizes": ps_vals,                "results": to_list(res3)},
    }
    with open("q2_decode_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved → q2_decode_results.json")
