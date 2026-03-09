"""Profile GPU kernel breakdown for our chunked engine."""
from __future__ import annotations
import gc
from collections import defaultdict

import torch
from torch.profiler import profile, ProfilerActivity
from transformers import AutoTokenizer

WEIGHT_PATH = "/local1/cse554/models/meta-llama/Llama-3.2-1B"
NUM_REQUESTS = 20
INPUT_LEN = 512
OUTPUT_LEN = 512
TOKEN_BUDGET = 8192
SEED = 554


def categorize_kernel(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ["flashinfer", "attention", "flash_attn", "sdpa",
                             "batch_prefill", "batch_decode"]):
        return "Attention"
    if any(k in n for k in ["rope", "rotary", "apply_rope"]):
        return "RoPE"
    if any(k in n for k in ["gemm", "matmul", "cublas", "cutlass", "mm_",
                             "ampere_fp16", "sm80_xmma", "sm75_xmma"]):
        return "MatMul (FFN + projections)"
    if any(k in n for k in ["silu", "gelu", "relu", "activation"]):
        return "Activations"
    if any(k in n for k in ["rmsnorm", "layernorm", "rms_norm", "layer_norm"]):
        return "LayerNorm"
    if any(k in n for k in ["elementwise", "add_kernel", "vectorized",
                             "copy", "memcpy", "memset", "fill"]):
        return "Elementwise / Memory"
    if any(k in n for k in ["embedding", "index_select", "gather"]):
        return "Embedding"
    if any(k in n for k in ["argmax", "topk", "sample", "softmax"]):
        return "Sampling"
    return "Other"


def summarize_profile(prof, label: str):
    events = prof.key_averages()
    cat_time = defaultdict(float)
    for evt in events:
        if evt.device_time_total > 0:
            cat = categorize_kernel(evt.key)
            cat_time[cat] += evt.device_time_total / 1000.0

    total = sum(cat_time.values())
    print(f"\n{'=' * 60}")
    print(f"  {label} — GPU kernel breakdown")
    print(f"  Total GPU time: {total:.1f} ms")
    print(f"{'=' * 60}")
    for cat, ms in sorted(cat_time.items(), key=lambda x: -x[1]):
        pct = ms / total * 100 if total > 0 else 0
        print(f"  {cat:<35s} {ms:>8.1f} ms  ({pct:>5.1f}%)")
    print()


def make_templates():
    tokenizer = AutoTokenizer.from_pretrained(WEIGHT_PATH)
    templates = []
    for i in range(NUM_REQUESTS):
        torch.manual_seed(SEED + i)
        ids = torch.randint(0, tokenizer.vocab_size, (INPUT_LEN,), dtype=torch.int64)
        ids[0] = tokenizer.bos_token_id or 0
        templates.append((ids, OUTPUT_LEN))
    return templates, tokenizer


def profile_ours(templates, tokenizer):
    from chunked_engine import Engine as ChunkedEngine
    from chunked_scheduler import Scheduler as ChunkedScheduler, InputRequest

    engine = ChunkedEngine()
    scheduler = ChunkedScheduler(engine, token_batch_size=TOKEN_BUDGET)
    for ids, olen in templates:
        text = tokenizer.decode(ids, skip_special_tokens=False)
        scheduler.add_req(InputRequest(text, olen))

    for _ in range(3):
        if not scheduler.finished():
            scheduler.run()

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        while not scheduler.finished():
            scheduler.run()

    del engine, scheduler
    gc.collect(); torch.cuda.empty_cache()
    return prof


if __name__ == "__main__":
    templates, tokenizer = make_templates()

    print("Profiling our chunked engine...")
    prof_ours = profile_ours(templates, tokenizer)
    summarize_profile(prof_ours, "Our Chunked Engine")
