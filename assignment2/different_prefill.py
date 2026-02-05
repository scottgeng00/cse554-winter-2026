from numpy import concat, require
import torch
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import sys
sys.path.append("../")  # Adjust the path to import the helper module
from helper import WeightManager, apply_rope, extract_model_weights
from uniform_prefill import apply_rope_batched  # dont need to change this fn for different length case.


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
    
    def run(self, input_ids, prefill = True):
        bsz = len(input_ids)

        # left-pad the input ids to the same length, and keep track of attention mask.
        # use left padding so the next generated token is at the end of the sequence.
        max_len = max(seq.shape[0] for seq in input_ids)
        seq_lens = torch.tensor([seq.shape[0] for seq in input_ids], device='cuda')
        padding_lens = max_len - seq_lens
        for i, seq in enumerate(input_ids):
            if seq.shape[0] < max_len:
                input_ids[i] = torch.nn.functional.pad(seq, (padding_lens[i], 0), mode='constant', value=0)
        input_ids = torch.stack(input_ids)
        
        attention_mask = torch.ones(bsz, max_len, device='cuda')
        for i in range(bsz):
            attention_mask[i, :padding_lens[i]] = 0

        # when we have prefill, we need to fill out everything.
        # otherwise, we use the kv cache.
        input_tensor = torch.tensor(input_ids, dtype=torch.int32, device='cuda')
        if prefill:
            hidden_state = self.weights["embedding"][input_tensor]
        else:
            # batched case - just handle batch dimension.
            hidden_state = self.weights["embedding"][input_tensor[:, -1:]]

        for current_layer in range(self.layers):
            rms = torch.sqrt(torch.mean(hidden_state ** 2, dim=-1, keepdim=True) + 1e-5)
            normalized_x = hidden_state / rms
            x = normalized_x * self.weights["layernormAttn_weight"][current_layer]

            q = x.matmul(self.weights["self_attn_q_proj_weight"][current_layer].t())
            k = x.matmul(self.weights["self_attn_k_proj_weight"][current_layer].t())
            v = x.matmul(self.weights["self_attn_v_proj_weight"][current_layer].t())

            if prefill:
                apply_rope_batched(q, output=q, head_dim=self.head_dim, offset=padding_lens)
                apply_rope_batched(k, output=k, head_dim=self.head_dim, offset=padding_lens)
            else:
                apply_rope_batched(q, output=q, head_dim=self.head_dim, offset=self.kv_cache[current_layer][0].shape[1])
                apply_rope_batched(k, output=k, head_dim=self.head_dim, offset=self.kv_cache[current_layer][0].shape[1])

            if prefill:
                # save to kv-cache
                self.kv_cache[current_layer] = (k, v)
            else:
                # load from kv-cache
                k_prev = self.kv_cache[current_layer][0]
                v_prev = self.kv_cache[current_layer][1]
                # concat the new k and v to the previous k and v
                # batched case - concat at sequence dim, batch dim is the same.
                k = torch.cat([k_prev, k], dim=1)
                v = torch.cat([v_prev, v], dim=1)
                # re-save to kv-cache
                self.kv_cache[current_layer] = (k, v)

            sub_q = q.view(bsz, -1, self.num_qo_heads, self.head_dim)
            sub_k = k.view(bsz, -1, self.num_kv_heads, self.head_dim)
            sub_v = v.view(bsz, -1, self.num_kv_heads, self.head_dim)

            scale = 1.0 / (self.head_dim ** 0.5)
            group_size = self.num_qo_heads // self.num_kv_heads
            n_q = sub_q.shape[1]
            n_k = sub_k.shape[1]

            sub_k = sub_k.repeat_interleave(group_size, dim=2)
            sub_v = sub_v.repeat_interleave(group_size, dim=2)

            sub_q_t = sub_q.permute(0, 2, 1, 3)
            sub_k_t = sub_k.permute(0, 2, 1, 3)

            scores = torch.matmul(sub_q_t, sub_k_t.transpose(-2, -1)) * scale

            if prefill:
                causal_mask = torch.tril(torch.ones(n_q, n_k, dtype=torch.bool, device=scores.device))
                key_valid = attention_mask.bool()[:, None, None, :] # [bsz, 1, 1, max_len]
                # combined: causal allowed AND key is real (not padding)
                combined_mask = causal_mask[None, None, :, :] & key_valid 
                # non valid positions should be masked to -inf
                scores = scores.masked_fill(~combined_mask, float("-inf"))
            # we dont need to mask for non-prefill since we are only producing one token.

            attn_weights = torch.softmax(scores, dim=-1)
            v_t = sub_v.permute(0, 2, 1, 3)
            attn_output = torch.matmul(attn_weights, v_t)

            attn_output = attn_output.permute(0, 2, 1, 3)
            attn_output = attn_output.reshape(bsz, -1, self.num_qo_heads * self.head_dim)

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
        sample_output = torch.argmax(logits, dim=-1)
        # generate_batched expects cpu tensors
        return sample_output[:, -1].to('cpu')
    
    def generate_batched(self, input_string, rounds=20):
        input_ids_list = []
        for input_string in input_string:
            input_ids = self.tokenizer(input_string, return_tensors="pt").input_ids[0]
            input_ids_list.append(input_ids)
            
        output_ids_list = input_ids_list  
        new_token = self.run(input_ids_list)
        for i in range(len(input_ids_list)):
            output_ids_list[i] = torch.cat((output_ids_list[i], new_token[i:i+1]), dim=0)

        for round in range(rounds - 1):
            print(f"Round {round}")
            input_ids_list = []
            for output_ids in output_ids_list:
                input_ids_list.append(output_ids[-1:])
            new_token = self.run(input_ids_list, prefill=False)
            
            for i in range(len(input_ids_list)):
                output_ids_list[i] = torch.cat((output_ids_list[i], new_token[i:i+1]), dim=0)
        output_text_list = []
        for output_ids in output_ids_list:
            output_text_list.append(self.tokenizer.decode(output_ids, skip_special_tokens=True))
        return output_text_list

########################################
# Benchmarking Function
########################################
def benchmark_and_plot():
    """
    Benchmark generation throughput across different batch sizes.
    Input length: 512 tokens, Output length: 128 tokens
    Batch sizes: 2^0 to 2^6 (1, 2, 4, 8, 16, 32, 64)
    """
    input_length = 512
    output_length = 128
    batch_sizes = [2**i for i in range(7)]  # 1, 2, 4, 8, 16, 32, 64
    
    # Load model once (outside timing)
    print("Loading model...")
    engine = Engine()
    
    # Warmup
    print("Warming up...")
    dummy_ids = [torch.ones(10, dtype=torch.int64) for _ in range(2)]
    _ = engine.run(dummy_ids)
    engine.kv_cache = {}
    torch.cuda.synchronize()
    
    generation_times = []
    throughputs = []
    
    for batch_size in batch_sizes:
        print(f"\nBenchmarking batch size: {batch_size}")
        
        # Create synthetic input: list of [input_length] tensors
        # Using token ID 1 (common token) repeated
        input_ids_list = [torch.ones(input_length, dtype=torch.int64) for _ in range(batch_size)]
        
        # Clear KV cache
        engine.kv_cache = {}
        
        # Synchronize before timing
        torch.cuda.synchronize()
        
        # Start timing with CUDA events
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        
        # Prefill
        new_token = engine.run(input_ids_list, prefill=True)
        
        # Append tokens to track output
        output_ids_list = [torch.cat([inp, new_token[i:i+1]]) for i, inp in enumerate(input_ids_list)]
        
        # Decode loop
        for _ in range(output_length - 1):
            decode_input = [ids[-1:] for ids in output_ids_list]
            new_token = engine.run(decode_input, prefill=False)
            for i in range(batch_size):
                output_ids_list[i] = torch.cat([output_ids_list[i], new_token[i:i+1]])
        
        end_event.record()
        torch.cuda.synchronize()
        
        # Calculate metrics
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
        total_tokens_generated = batch_size * output_length
        throughput = total_tokens_generated / elapsed_time
        
        generation_times.append(elapsed_time)
        throughputs.append(throughput)
        
        print(f"  Time: {elapsed_time:.3f}s")
        print(f"  Tokens generated: {total_tokens_generated}")
        print(f"  Throughput: {throughput:.2f} tokens/s")
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Generation time vs batch size
    axes[0].plot(batch_sizes, generation_times, 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Batch Size', fontsize=12)
    axes[0].set_ylabel('Generation Time (seconds)', fontsize=12)
    axes[0].set_title('Generation Time vs Batch Size', fontsize=14)
    axes[0].set_xscale('log', base=2)
    axes[0].set_xticks(batch_sizes)
    axes[0].set_xticklabels([str(bs) for bs in batch_sizes])
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Throughput vs batch size
    axes[1].plot(batch_sizes, throughputs, 's-', linewidth=2, markersize=8, color='green')
    axes[1].set_xlabel('Batch Size', fontsize=12)
    axes[1].set_ylabel('Throughput (tokens/s)', fontsize=12)
    axes[1].set_title('Throughput vs Batch Size', fontsize=14)
    axes[1].set_xscale('log', base=2)
    axes[1].set_xticks(batch_sizes)
    axes[1].set_xticklabels([str(bs) for bs in batch_sizes])
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('assignment2_figs/different_prefill_benchmark.pdf', dpi=150)
    
    return batch_sizes, generation_times, throughputs


########################################
# Main Loop: Text Generation
########################################
if __name__ == "__main__":
    input_string = "Hi, who are you?"
    input_string_list = [input_string for _ in range(10)]
    another_input_string = "The University of Washington is located in"
    for _ in range(10):
        input_string_list.append(another_input_string)
    engine = Engine()
    output_text = engine.generate_batched(input_string_list, rounds=20)
    print("Generated Text:", output_text)
    # benchmarking
    benchmark_and_plot()