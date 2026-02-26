import numpy as np
import matplotlib.pyplot as plt
import json

# Reference config for each model
llama3_1b_config = {
    "hidden_size": 2048,
    "num_attention_heads": 32,
    "num_key_value_heads": 8
}

llama3_3b_config = {
    "hidden_size": 3072,
    "num_attention_heads": 24,
    "num_key_value_heads": 8
}

llama3_8b_config = {
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 8
}

# Sequence lengths (powers of 2)
p_llama3 = 2 ** np.arange(7, 16)   # 2^7 to 2^15

# Load real benchmark data
with open("q2_prefill_results.json") as f:
    prefill_data = json.load(f)
with open("q2_decode_results.json") as f:
    decode_data = json.load(f)

models = ['LLaMA3-1B', 'LLaMA3-3B', 'LLaMA3-8B']

# ── Prefill: bs=1, varying p ──
res = prefill_data["exp1_vary_p"]["results"]
fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
for i, name in enumerate(models):
    axs[i].plot(p_llama3, res[name]["sdpa"], label='PyTorch SDPA', marker='o')
    axs[i].plot(p_llama3, res[name]["flashinfer"], label='FlashInfer', marker='x')
    axs[i].set_xscale('log', base=2)
    axs[i].set_title(name)
    axs[i].set_xlabel('p (sequence length)')
    axs[i].set_xticks(p_llama3)
    axs[i].set_xticklabels([str(p) for p in p_llama3])
    axs[i].legend()
    axs[i].grid(True, which='both')
axs[0].set_ylabel('Compute Utilization (TFLOPs)')
fig.suptitle('Prefill Attention — bs=1, varying p', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('q2_prefill_vary_p.png', dpi=300)
print("Saved → q2_prefill_vary_p.png")

# ── Prefill: varying batch size, p=1024 ──
res = prefill_data["exp2_vary_bs"]["results"]
bs_values = np.array(prefill_data["exp2_vary_bs"]["bs_values"])
fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
for i, name in enumerate(models):
    axs[i].plot(bs_values, res[name]["sdpa"], label='PyTorch SDPA', marker='o')
    axs[i].plot(bs_values, res[name]["flashinfer"], label='FlashInfer', marker='x')
    axs[i].set_xscale('log', base=2)
    axs[i].set_title(name)
    axs[i].set_xlabel('Batch Size')
    axs[i].set_xticks(bs_values)
    axs[i].set_xticklabels([str(b) for b in bs_values])
    axs[i].legend()
    axs[i].grid(True, which='both')
axs[0].set_ylabel('Compute Utilization (TFLOPs)')
fig.suptitle('Prefill Attention — p=1024, varying batch size', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('q2_prefill_vary_bs.png', dpi=300)
print("Saved → q2_prefill_vary_bs.png")

# ── Decode: bs=1, varying c ──
res = decode_data["exp1_vary_c"]["results"]
c_values = np.array(decode_data["exp1_vary_c"]["c_values"])
fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
for i, name in enumerate(models):
    axs[i].plot(c_values, res[name]["sdpa"], label='PyTorch SDPA', marker='o')
    axs[i].plot(c_values, res[name]["flashinfer"], label='FlashInfer', marker='x')
    axs[i].set_xscale('log', base=2)
    axs[i].set_title(name)
    axs[i].set_xlabel('c (context length)')
    axs[i].set_xticks(c_values)
    axs[i].set_xticklabels([str(c) for c in c_values])
    axs[i].legend()
    axs[i].grid(True, which='both')
axs[0].set_ylabel('Memory Bandwidth (GB/s)')
fig.suptitle('Decode Attention — bs=1, varying c', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('q2_decode_vary_c.png', dpi=300)
print("Saved → q2_decode_vary_c.png")

# ── Decode: varying batch size, c=1024 ──
res = decode_data["exp2_vary_bs"]["results"]
bs_values = np.array(decode_data["exp2_vary_bs"]["bs_values"])
fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
for i, name in enumerate(models):
    axs[i].plot(bs_values, res[name]["sdpa"], label='PyTorch SDPA', marker='o')
    axs[i].plot(bs_values, res[name]["flashinfer"], label='FlashInfer', marker='x')
    axs[i].set_xscale('log', base=2)
    axs[i].set_title(name)
    axs[i].set_xlabel('Batch Size')
    axs[i].set_xticks(bs_values)
    axs[i].set_xticklabels([str(b) for b in bs_values])
    axs[i].legend()
    axs[i].grid(True, which='both')
axs[0].set_ylabel('Memory Bandwidth (GB/s)')
fig.suptitle('Decode Attention — c=1024, varying batch size', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('q2_decode_vary_bs.png', dpi=300)
print("Saved → q2_decode_vary_bs.png")

# ── Decode: bs=128, c=1024, varying page_size (FlashInfer only) ──
res = decode_data["exp3_vary_pagesize"]["results"]
page_sizes = decode_data["exp3_vary_pagesize"]["page_sizes"]
fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
for i, name in enumerate(models):
    axs[i].plot(page_sizes, res[name]["flashinfer"], label='FlashInfer',
                marker='s', color='tab:green')
    axs[i].set_title(name)
    axs[i].set_xlabel('Page Size')
    axs[i].set_xticks(page_sizes)
    axs[i].legend()
    axs[i].grid(True, which='both')
axs[0].set_ylabel('Memory Bandwidth (GB/s)')
fig.suptitle('Decode Attention — bs=128, c=1024, varying page size (FlashInfer)',
             fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('q2_decode_vary_pagesize.png', dpi=300)
print("Saved → q2_decode_vary_pagesize.png")
