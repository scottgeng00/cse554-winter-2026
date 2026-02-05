import torch
from transformers import AutoTokenizer
import sys
sys.path.append("../")  # Adjust the path to import the helper module
from helper import WeightManager, apply_rope, extract_model_weights
import matplotlib.pyplot as plt


class Engine:
    """
    A class to manage the generation engine.
    """
    def __init__(self):
        ########################################
        # Model Configuration Parameters
        ########################################
        self.weight_path = "/local1/cse554/models/meta-llama/Llama-3.2-1B"
        self.head_dim = 64         # Dimensionality of each attention head
        self.num_qo_heads = 32      # Total number of query/output heads
        self.num_kv_heads = 8       # Total number of key/value heads
        self.layers = 16            # Number of transformer layers

        # Load the tokenizer for text processing
        # self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.weight_path,
            local_files_only=True
        )

        # Initialize and load model weights using the helper module
        weight_manager = WeightManager()
        weight_manager.load_from_safe_tensor(self.weight_path)

        # Extract all required model weights from the weight_map
        self.weights = extract_model_weights(weight_manager.weight_map, self.layers)
        
        self.kv_cache = {}
    
    def run(self, input_ids, prefill = True,):
        # basic code from lecture.

        # when we have prefill, we need to fill out everything.
        # otherwise, we use the kv cache.
        input_tensor = torch.tensor(input_ids, dtype=torch.int32, device='cuda')
        if prefill:
            hidden_state = self.weights["embedding"][input_tensor]
        else:
            hidden_state = self.weights["embedding"][input_tensor[-1:]]

        for current_layer in range(self.layers):
            rms = torch.sqrt(torch.mean(hidden_state ** 2, dim=-1, keepdim=True) + 1e-5)
            normalized_x = hidden_state / rms
            x = normalized_x * self.weights["layernormAttn_weight"][current_layer]

            q = x.matmul(self.weights["self_attn_q_proj_weight"][current_layer].t())
            k = x.matmul(self.weights["self_attn_k_proj_weight"][current_layer].t())
            v = x.matmul(self.weights["self_attn_v_proj_weight"][current_layer].t())

            if prefill:
                apply_rope(q, output=q, head_dim=self.head_dim, offset=0)
                apply_rope(k, output=k, head_dim=self.head_dim, offset=0)
            else:
                apply_rope(q, output=q, head_dim=self.head_dim, offset=len(self.kv_cache[current_layer][0]))
                apply_rope(k, output=k, head_dim=self.head_dim, offset=len(self.kv_cache[current_layer][0]))

            if prefill:
                # save to kv-cache
                self.kv_cache[current_layer] = (k, v)
            else:
                # load from kv-cache
                k_prev = self.kv_cache[current_layer][0]
                v_prev = self.kv_cache[current_layer][1]
                # concat the new k and v to the previous k and v
                k = torch.cat([k_prev, k], dim=0)
                v = torch.cat([v_prev, v], dim=0)
                # re-save to kv-cache
                self.kv_cache[current_layer] = (k, v)

            sub_q = q.view(-1, self.num_qo_heads, self.head_dim)
            sub_k = k.view(-1, self.num_kv_heads, self.head_dim)
            sub_v = v.view(-1, self.num_kv_heads, self.head_dim)

            scale = 1.0 / (self.head_dim ** 0.5)
            group_size = self.num_qo_heads // self.num_kv_heads
            n_q = sub_q.shape[0]
            n_k = sub_k.shape[0]

            sub_k = sub_k.repeat_interleave(group_size, dim=1)
            sub_v = sub_v.repeat_interleave(group_size, dim=1)

            sub_q_t = sub_q.permute(1, 0, 2)
            sub_k_t = sub_k.permute(1, 0, 2)

            scores = torch.matmul(sub_q_t, sub_k_t.transpose(-2, -1)) * scale

            if prefill:
                causal_mask = torch.tril(torch.ones(n_q, n_k, dtype=torch.bool, device=scores.device))
                scores = scores.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))
            # we dont need to mask for non-prefill since we are only producing one token.

            attn_weights = torch.softmax(scores, dim=-1)
            v_t = sub_v.permute(1, 0, 2)
            attn_output = torch.matmul(attn_weights, v_t)

            attn_output = attn_output.permute(1, 0, 2)
            attn_output = attn_output.reshape(-1, self.num_qo_heads * self.head_dim)

            o_proj_residual = attn_output.matmul(self.weights["o_proj_weight"][current_layer].t()) + hidden_state

            rms = torch.sqrt(torch.mean(o_proj_residual ** 2, dim=-1, keepdim=True) + 1e-5)
            normalized_x = o_proj_residual / rms
            layernormFFN_output = normalized_x.to(torch.float16) * self.weights["layernormFFN_weight"][current_layer]
            up_proj_output = layernormFFN_output.matmul(self.weights["up_proj_weight"][current_layer].t())
            gate_proj_output = layernormFFN_output.matmul(self.weights["gate_proj_weight"][current_layer].t())
            activation_output = up_proj_output * torch.nn.functional.silu(gate_proj_output)
            down_proj_output = activation_output.matmul(self.weights["down_proj_weight"][current_layer].t())
            hidden_state = down_proj_output + o_proj_residual

        rms = torch.sqrt(torch.mean(hidden_state ** 2, dim=-1, keepdim=True) + 1e-5)
        normalized_x = hidden_state / rms
        model_output = normalized_x.to(torch.float16) * self.weights["model_layernorm_weight"]

        logits = model_output.matmul(self.weights["lm_head_weight"].t())
        sample_output = torch.argmax(logits, dim=1)
        return sample_output[-1].item()
    
    def generate(self, input_string, rounds=20):
        input_ids = self.tokenizer.encode(input_string)

        print("Token IDs:", input_ids)
        output_ids = input_ids.copy()

        new_token = self.run(output_ids)
        output_ids.append(new_token)

        for round in range(rounds - 1):
            print(f"Round {round}")
            new_token = self.run(output_ids[-1:], prefill=False)
            output_ids.append(new_token)

        output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return output_text

########################################
# Main Loop: Text Generation
########################################
def benchmarking_and_plotting():
    # Output lengths to test
    output_lengths = list(range(128, 2049, 128))
    input_length = 1024
    
    # Create synthetic input tokens (use token ID 1 repeated)
    input_ids = [1] * input_length
    
    # Load model once (outside of timing)
    kv_engine = Engine()
    
    # Warm up GPU
    print("Warming up GPU...")
    _ = kv_engine.run(input_ids[:10])
    kv_engine.kv_cache = {}
    torch.cuda.synchronize()
    
    kv_times = []
    no_kv_times = []
    
    for output_len in output_lengths:
        print(f"\nBenchmarking output length: {output_len}")
        
        # ========== Benchmark WITH KV cache ==========
        kv_engine.kv_cache = {}
        
        # Use CUDA events for accurate timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        output_ids = input_ids.copy()
        
        torch.cuda.synchronize()
        start_event.record()
        
        # Prefill
        new_token = kv_engine.run(output_ids, prefill=True)
        output_ids.append(new_token)
        
        # Decode with KV cache
        for _ in range(output_len - 1):
            new_token = kv_engine.run(output_ids[-1:], prefill=False)
            output_ids.append(new_token)
        
        end_event.record()
        torch.cuda.synchronize()
        
        kv_time = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
        kv_times.append(kv_time)
        print(f"  KV cache time: {kv_time:.3f}s")
        
        # ========== Benchmark WITHOUT KV cache ==========
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        output_ids = input_ids.copy()
        
        torch.cuda.synchronize()
        start_event.record()
        
        # Generate without KV cache (recompute everything each step)
        # we can just use prefill=True to force this!
        for _ in range(output_len):
            new_token = kv_engine.run(output_ids, prefill=True)
            output_ids.append(new_token)
        
        end_event.record()
        torch.cuda.synchronize()
        
        no_kv_time = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
        no_kv_times.append(no_kv_time)
        print(f"  No KV cache time: {no_kv_time:.3f}s")
        print(f"  Speedup: {no_kv_time / kv_time:.2f}x")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(output_lengths, no_kv_times, 'o-', label='Without KV Cache', linewidth=2, markersize=6)
    plt.plot(output_lengths, kv_times, 's-', label='With KV Cache', linewidth=2, markersize=6)
    
    plt.xlabel('Output Length (tokens)', fontsize=12)
    plt.ylabel('Generation Time (seconds)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('assignment2_figs/kv_cache_benchmark.pdf', dpi=150)
    
    return output_lengths, kv_times, no_kv_times


if __name__ == "__main__":
    input_string = "Hi, who are you?"
    engine = Engine()
    output_text = engine.generate(input_string, rounds=50)
    print("Generated Text:", output_text)
    # benchmarking
    benchmarking_and_plotting()