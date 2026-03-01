import sys
from pathlib import Path
from typing import Dict, List

import torch
import flashinfer
from transformers import AutoTokenizer
import time
import gc
from tqdm import tqdm
import matplotlib.pyplot as plt

from flashinfer_pipeline import Engine

def make_fake_prompt(batch_size: int, seq_len: int, vocab_size: int) -> torch.Tensor:
    return [torch.randint(0, vocab_size, (seq_len,)) for _ in range(batch_size)]

def run_exp(engine: Engine, batch_size, prefill_len, decode_len, num_trials=3, warmup_trials=1) -> Dict[str, float]:
    all_times = []
    for trial in tqdm(range(num_trials + warmup_trials), desc=f"Running warmup + trials for batch_size={batch_size}, prefill_len={prefill_len}, decode_len={decode_len}"):
        engine.reset()
        prompt_ids = make_fake_prompt(batch_size, prefill_len, len(engine.tokenizer))
        prefill_time, decode_times, total_time = engine.generate_batched_from_ids_with_timings(
            prompt_ids, rounds=decode_len
        )
        if trial >= warmup_trials:  # Skip warmup trials
            all_times.append({
                "prefill_time": prefill_time,
                "decode_time": sum(decode_times.values()),
                "total_time": total_time
            })
    
    
    avg_prefill_time = sum(t["prefill_time"] for t in all_times) / num_trials
    avg_decode_time = sum(t["decode_time"] for t in all_times) / num_trials
    avg_total_time = sum(t["total_time"] for t in all_times) / num_trials

    print(f"Batch Size: {batch_size}, Prefill Len: {prefill_len}, Decode Len: {decode_len} ({num_trials} trials)")
    print(f"Prefill Time: {avg_prefill_time:.4f} sec, Decode Time: {avg_decode_time:.4f} sec, Total Time: {avg_total_time:.4f} sec\n")

    return {
        "avg_prefill_time": avg_prefill_time,
        "avg_decode_time": avg_decode_time,
        "avg_total_time": avg_total_time
    }


def exp1(engine: Engine):
    batch_size = 32
    prefill_length = 256
    decode_lengths = [2**i for i in range(5, 11)]
    num_trials = 5

    results = dict()
    for decode_len in decode_lengths:
        results[decode_len] = run_exp(engine, batch_size, prefill_length, decode_len, num_trials=num_trials, warmup_trials=1)

    # Plot end-to-end time curve with the log(decode length) as the x-axis.
    fig, ax = plt.subplots()
    ax.plot(decode_lengths, [results[dl]["avg_total_time"] for dl in decode_lengths], marker='o')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Decode Length (log scale)')
    ax.set_ylabel(f'Average Total Time over {num_trials} Trials (sec)')
    ax.set_title(f'End-to-End Time vs Decode Length (Batch Size={batch_size}, Prefill Length={prefill_length})')
    fig.savefig(f'exp1.png', dpi=300)



def exp2(engine: Engine):
    batch_size = 1
    prefill_lengths = [2**i for i in range(8, 15)]
    num_trials = 5

    results = dict()
    for prefill_len in prefill_lengths:
        results[prefill_len] = run_exp(engine, batch_size, prefill_len, 1, num_trials=num_trials, warmup_trials=1)

    # Plot prefill time curve with the log(prefill length) as the x-axis.
    fig, ax = plt.subplots()
    ax.plot(prefill_lengths, [results[pl]["avg_prefill_time"] for pl in prefill_lengths], marker='o')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Prefill Length (log scale)')
    ax.set_ylabel(f'Average Prefill Time over {num_trials} Trials (sec)')
    ax.set_title(f'Prefill Time vs Prefill Length (Batch Size={batch_size}, Decode Length=1)')

    fig.savefig(f'exp2.png', dpi=300)


if __name__ == "__main__":
    engine = Engine()
    
    # exp1(engine)

    exp2(engine)