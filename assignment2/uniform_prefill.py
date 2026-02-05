import torch
from transformers import AutoTokenizer
import sys
sys.path.append("../")  # Adjust the path to import the helper module
from helper import WeightManager, extract_model_weights, rotate_half


def apply_rope_batched(x: torch.Tensor, output: torch.Tensor, head_dim: int, offset: int = 0) -> None:
    """
    Applies RoPE (Rotary Positional Embedding) to the input tensor for batched case.
    
    RoPE adds position-dependent rotations to the tensor.
    
    Args:
        x (torch.Tensor): Input tensor with shape [bsz, seq_len, head_dim].
        output (torch.Tensor): Tensor to store the result (same shape as x).
        head_dim (int): Dimensionality of each attention head.
        offset (int | torch.Tensor): Positional offset. May be a scalar or a tensor of shape [bsz].
    """
    bsz, seq_len, _ = x.shape  # [bsz, seq_len, head_dim]
    device = x.device
    dtype = x.dtype

    # Create positions: shape [bsz, seq_len]
    if isinstance(offset, int):
        positions = torch.arange(offset, offset + seq_len, device=device, dtype=dtype).unsqueeze(0).expand(bsz, -1)
    else:
        # batched case - offset is a tensor of shape [bsz], may be diff offsets for each sequence.
        base_offsets = torch.arange(0, seq_len, device=device, dtype=dtype).unsqueeze(0).expand(bsz, -1)
        # just add the offsets
        positions = base_offsets + offset.unsqueeze(1)

    base = 500000.0
    # Compute inverse frequency: shape [head_dim/2]
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.int64)
                                 .float().to(device) / head_dim))
    
    # Expand dimensions for broadcasting:
    inv_freq_expanded = inv_freq[None, :, None].float().expand(bsz, -1, 1)
    position_ids_expanded = positions[:, None, :].float()
    
    # Compute frequency embeddings
    with torch.autocast(device_type=device.type, enabled=False):
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        # Duplicate frequencies for cos and sin parts:
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
    
    # Reshape for multi-head compatibility:
    cos = cos.unsqueeze(1)  # [bsz, 1, seq_len, head_dim]
    sin = sin.unsqueeze(1)  # [bsz, 1, seq_len, head_dim]

    # Reshape x for applying RoPE:
    x = x.reshape(bsz, seq_len, -1, head_dim).transpose(1, 2)
    # Apply RoPE rotation
    x_rotated = x * cos + rotate_half(x) * sin
    # Restore original shape and copy into output
    output.copy_(x_rotated.transpose(1, 2).reshape(bsz, seq_len, -1).to(dtype=dtype))

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
        # basic code from lecture.
        bsz = input_ids.shape[0]

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
                apply_rope_batched(q, output=q, head_dim=self.head_dim, offset=0)
                apply_rope_batched(k, output=k, head_dim=self.head_dim, offset=0)
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
                scores = scores.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))
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
        return sample_output[:, -1].unsqueeze(1).to('cpu')
    
    def generate_batched(self, input_string, rounds=20):
        input_ids_list = self.tokenizer(input_string, return_tensors="pt", padding=False).input_ids
        print("Input String:", input_string)

        print("Token IDs:", input_ids_list)
        output_ids_list = input_ids_list  

        new_token = self.run(output_ids_list)
        print("New Token Shape:", new_token.shape)
        output_ids_list = torch.cat((output_ids_list, new_token), dim=1)

        for round in range(rounds - 1):
            print(f"Round {round}")
            new_token = self.run(output_ids_list[:, -1:], prefill=False)
            output_ids_list = torch.cat((output_ids_list, new_token), dim=1)

        output_text = self.tokenizer.batch_decode(output_ids_list, skip_special_tokens=True)
        return output_text

########################################
# Main Loop: Text Generation
########################################
if __name__ == "__main__":
    input_string = "Hi, who are you?"
    input_string_list = [input_string for _ in range(10)]
    another_input_string = "Hi, how are you?"
    for _ in range(10):
        input_string_list.append(another_input_string)
    engine = Engine()
    output_text = engine.generate_batched(input_string_list, rounds=20)
    print("Generated Text:", output_text)