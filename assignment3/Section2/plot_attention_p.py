import numpy as np
import matplotlib.pyplot as plt

#Reference config for each model
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

# Fake TFLOPs data generator
def fake_tflops(seq_lens, model_factor):
    return np.log2(seq_lens) * model_factor + np.random.normal(0, 0.5, size=len(seq_lens))

# Generate fake compute utilization data
llama3_1b_sdpa = fake_tflops(p_llama3, 2.0)
llama3_1b_flashinfer = fake_tflops(p_llama3, 2.2)

llama3_3b_sdpa = fake_tflops(p_llama3, 2.5)
llama3_3b_flashinfer = fake_tflops(p_llama3, 2.8)

llama3_8b_sdpa = fake_tflops(p_llama3, 3.5)
llama3_8b_flashinfer = fake_tflops(p_llama3, 3.9)

# Plotting setup
fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
models = ['LLaMA3-1B', 'LLaMA3-3B', 'LLaMA3-8B']

# LLaMA2-7B plot
axs[0].plot(p_llama3, llama3_1b_sdpa, label='PyTorch SDPA', marker='o')
axs[0].plot(p_llama3, llama3_1b_flashinfer, label='FlashInfer', marker='x')
axs[0].set_xscale('log', base=2)
axs[0].set_title(models[0])
axs[0].set_xlabel('p (sequence length)')
axs[0].set_ylabel('Compute Utilization (TFLOPs)')
axs[0].set_xticks(p_llama3)
axs[0].set_xticklabels([str(p) for p in p_llama3])
axs[0].legend()
axs[0].grid(True, which='both')

# LLaMA3-8B plot
axs[1].plot(p_llama3, llama3_3b_sdpa, label='PyTorch SDPA', marker='o')
axs[1].plot(p_llama3, llama3_3b_flashinfer, label='FlashInfer', marker='x')
axs[1].set_xscale('log', base=2)
axs[1].set_title(models[1])
axs[1].set_xlabel('p (sequence length)')
axs[1].set_xticks(p_llama3)
axs[1].set_xticklabels([str(p) for p in p_llama3])
axs[1].legend()
axs[1].grid(True, which='both')

# LLaMA3-70B plot
axs[2].plot(p_llama3, llama3_8b_sdpa, label='PyTorch SDPA', marker='o')
axs[2].plot(p_llama3, llama3_8b_flashinfer, label='FlashInfer', marker='x')
axs[2].set_xscale('log', base=2)
axs[2].set_title(models[2])
axs[2].set_xlabel('p (sequence length)')
axs[2].set_xticks(p_llama3)
axs[2].set_xticklabels([str(p) for p in p_llama3])
axs[2].legend()
axs[2].grid(True, which='both')

# Overall figure title and layout
fig.suptitle('Prefill Attention Compute Utilization (Fake Data)', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('attention_compute_utilization.png', dpi=300)
