from __future__ import annotations

import argparse
import gc
import random
import time
from collections import deque

import numpy as np
import torch

from continous_engine import Engine as ContEngine, Request as ContRequest
from continous_scheduler import Scheduler as ContScheduler, InputRequest as ContInputRequest
from chunked_engine import Engine as ChunkedEngine
from chunked_scheduler import Scheduler as ChunkedScheduler, InputRequest as ChunkedInputRequest


# ---------------------------------------------------------------------------
#  Workload generation
# ---------------------------------------------------------------------------

def generate_workload_uniform(num_requests, in_min, in_max, out_min, out_max, seed):
    rng = random.Random(seed)
    return [(rng.randint(in_min, in_max), rng.randint(out_min, out_max))
            for _ in range(num_requests)]


def generate_workload_lognormal(num_requests, in_mean, in_sigma, out_min, out_max, seed):
    rng = np.random.RandomState(seed)
    input_lens = np.clip(np.round(rng.lognormal(in_mean, in_sigma, size=num_requests)).astype(int), 1, None)
    output_lens = rng.randint(out_min, out_max + 1, size=num_requests)
    return list(zip(input_lens.tolist(), output_lens.tolist()))


def make_templates(engine, workload, seed):
    vocab_size = int(engine.weights["embedding"].size(0))
    bos = engine.tokenizer.bos_token_id or 0
    templates = []
    for i, (ilen, olen) in enumerate(workload):
        torch.manual_seed(seed + i)
        ids = torch.randint(0, vocab_size, (ilen,), dtype=torch.int64)
        ids[0] = bos
        templates.append((ids, olen))
    return templates


# ---------------------------------------------------------------------------
#  Naive & continuous use the continuous engine directly
# ---------------------------------------------------------------------------

def _to_requests(templates):
    return [ContRequest(i, ids.clone(), olen) for i, (ids, olen) in enumerate(templates)]

def _step(active, new_tokens):
    for req, tok in zip(active, new_tokens):
        req.output_token_ids = torch.cat([req.output_token_ids, tok.view(1)])

def _drop_done(engine, reqs):
    keep = []
    for r in reqs:
        if (r.current_length - r.prompt_length) >= r.output_length:
            c = engine.kv_cache_map.pop(r.request_id, None)
            if c:
                c.release()
        else:
            keep.append(r)
    return keep


def bench_naive(templates, bs, record_iters=False):
    engine = ContEngine()
    reqs = _to_requests(templates)
    iter_times = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(0, len(reqs), bs):
        batch = reqs[i:i + bs]
        if not batch:
            continue
        torch.cuda.synchronize(); it0 = time.perf_counter()
        _step(batch, engine.run(batch, num_decode_req=0))
        batch = _drop_done(engine, batch)
        torch.cuda.synchronize()
        if record_iters:
            iter_times.append(time.perf_counter() - it0)
        while batch:
            torch.cuda.synchronize(); it0 = time.perf_counter()
            _step(batch, engine.run(batch, num_decode_req=len(batch)))
            batch = _drop_done(engine, batch)
            torch.cuda.synchronize()
            if record_iters:
                iter_times.append(time.perf_counter() - it0)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    del engine
    gc.collect(); torch.cuda.empty_cache()
    return elapsed, iter_times


def bench_continuous(templates, bs, record_iters=False):
    engine = ContEngine()
    pending = deque(_to_requests(templates))
    decoding: list[ContRequest] = []
    iter_times = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    while pending or decoding:
        prefill = []
        while pending and len(decoding) + len(prefill) < bs:
            prefill.append(pending.popleft())
        active = decoding + prefill
        if not active:
            break
        torch.cuda.synchronize(); it0 = time.perf_counter()
        _step(active, engine.run(active, num_decode_req=len(decoding)))
        decoding = _drop_done(engine, decoding)
        prefill = _drop_done(engine, prefill)
        decoding.extend(prefill)
        torch.cuda.synchronize()
        if record_iters:
            iter_times.append(time.perf_counter() - it0)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    del engine
    gc.collect(); torch.cuda.empty_cache()
    return elapsed, iter_times


def bench_chunked(templates, tokenizer, token_budget, record_iters=False):
    engine = ChunkedEngine()
    scheduler = ChunkedScheduler(engine, token_batch_size=token_budget)
    for ids, olen in templates:
        text = tokenizer.decode(ids, skip_special_tokens=False)
        scheduler.add_req(ChunkedInputRequest(text, olen))
    iter_times = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    while not scheduler.finished():
        torch.cuda.synchronize(); it0 = time.perf_counter()
        scheduler.run()
        torch.cuda.synchronize()
        if record_iters:
            iter_times.append(time.perf_counter() - it0)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    del engine, scheduler
    gc.collect(); torch.cuda.empty_cache()
    return elapsed, iter_times


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Profile naive / continuous / chunked scheduling")
    p.add_argument("--methods", nargs="+", default=["naive", "continuous", "chunked"],
                   choices=["naive", "continuous", "chunked"])
    p.add_argument("--num-requests", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128,
                   help="Request batch size for naive/continuous")
    p.add_argument("--token-budget", type=int, default=512,
                   help="Token budget per iteration for chunked")

    p.add_argument("--input-dist", choices=["uniform", "lognormal"], default="uniform")
    p.add_argument("--min-input-len", type=int, default=1)
    p.add_argument("--max-input-len", type=int, default=10)
    p.add_argument("--input-mean", type=float, default=6.0,
                   help="μ for lognormal input distribution")
    p.add_argument("--input-sigma", type=float, default=0.7,
                   help="σ for lognormal input distribution")

    p.add_argument("--min-output-len", type=int, default=1)
    p.add_argument("--max-output-len", type=int, default=128)
    p.add_argument("--seed", type=int, default=554)

    p.add_argument("--csv", type=str, default=None,
                   help="Save per-iteration times to this CSV file")
    args = p.parse_args()

    if args.input_dist == "uniform":
        workload = generate_workload_uniform(
            args.num_requests, args.min_input_len, args.max_input_len,
            args.min_output_len, args.max_output_len, args.seed)
        input_desc = f"uniform[{args.min_input_len}, {args.max_input_len}]"
    else:
        workload = generate_workload_lognormal(
            args.num_requests, args.input_mean, args.input_sigma,
            args.min_output_len, args.max_output_len, args.seed)
        input_desc = f"lognormal(μ={args.input_mean}, σ={args.input_sigma})"

    in_lens = [w[0] for w in workload]
    out_lens = [w[1] for w in workload]
    print(f"Requests:       {args.num_requests}")
    print(f"Input lengths:  {input_desc}  "
          f"mean={np.mean(in_lens):.1f}, min={min(in_lens)}, max={max(in_lens)}")
    print(f"Output lengths: uniform[{args.min_output_len}, {args.max_output_len}]  "
          f"mean={np.mean(out_lens):.1f}")
    print(f"Methods:        {', '.join(args.methods)}")
    print()

    tmp_engine = ContEngine()
    templates = make_templates(tmp_engine, workload, args.seed)
    tokenizer = tmp_engine.tokenizer
    del tmp_engine
    gc.collect(); torch.cuda.empty_cache()

    record = args.csv is not None
    results = {}
    all_iter_times = {}

    for method in args.methods:
        if method == "naive":
            t, iters = bench_naive(templates, args.batch_size, record_iters=record)
            label = f"Naive (bs={args.batch_size})"
        elif method == "continuous":
            t, iters = bench_continuous(templates, args.batch_size, record_iters=record)
            label = f"Continuous (bs={args.batch_size})"
        elif method == "chunked":
            t, iters = bench_chunked(templates, tokenizer, args.token_budget, record_iters=record)
            label = f"Chunked (budget={args.token_budget})"
        results[method] = (label, t)
        all_iter_times[method] = (label, iters)
        print(f"  {label}: {t:.4f}s")
        if record and iters:
            ms = [x * 1000 for x in iters]
            print(f"    {len(iters)} iterations, mean={np.mean(ms):.2f}ms, max={np.max(ms):.2f}ms")

    print()
    print("=== Comparison ===")
    names = list(results.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            la, ta = results[a]
            lb, tb = results[b]
            if tb > 0:
                print(f"  {la} / {lb} = {ta / tb:.4f}x")

    if record:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["method", "iteration", "time_ms"])
            for method in args.methods:
                label, iters = all_iter_times[method]
                for i, t in enumerate(iters):
                    w.writerow([method, i, t * 1000])
        print(f"\nSaved iteration times to {args.csv}")


if __name__ == "__main__":
    main()
