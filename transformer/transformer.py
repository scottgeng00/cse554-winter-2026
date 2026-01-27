# 1-
import os
import torch
# https://huggingface.co/docs/transformers/v4.51.1/en/model_doc/auto#transformers.AutoTokenizer
from transformers import AutoTokenizer # We'll write most code not all, in particular we'll need stuff like word -> token transformations
from helper import WeightManager, apply_rope, extract_model_weights # not super important for understanding transformer internals
# Some things in the helper: applying rope is taking an embedding and modifying it to include positional information
# Skip to 2- , main function below
# TODO:rename query/output heads to attention heads 

########################################
# Model Configuration Parameters
########################################
weight_path = os.environ.get("TRANSFORMER_WEIGHT_PATH", "/data/Meta-Llama-3-8B-Instruct")
layers = 32 # Llama-3 8B has 32 layers.
head_dim = 128         # Dimensionality of each attention head, (hidden_dim / # of heads)
num_qo_heads = 32      # Total number of query/output heads (also known as attention heads)
num_kv_heads = 8       # Total number of key/value heads
# 4 query heads share 1 key/value head. This is called grouped query attention

# Load the tokenizer for text processing. Llama uses SentencePiece tokenizer,
# a variant of BPE (Byte Pair Encoding) tokenizer. The tokenizer learns subword
# units from the training corpus and uses them to tokenize input text.
# 
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

# 5- 
# Initialize and load model weights using the helper module
weight_manager = WeightManager()
# Safetensors are a serialization format for tensors that is more efficient and safer
# than traditional formats. Loading doesn't execute arbitrary code, making it safe
weight_manager.load_from_safe_tensor(weight_path)
# Extract model weights from the weight map
weights = extract_model_weights(weight_manager.weight_map, layers)

# Unpack weights for convenient reference
embedding = weights["embedding"] # token_ids -> embedding

# Attention weights, they are per layer. We'll index them by layer number
layernormAttn_weight = weights["layernormAttn_weight"]
self_attn_q_proj_weight = weights["self_attn_q_proj_weight"]
self_attn_k_proj_weight = weights["self_attn_k_proj_weight"]
self_attn_v_proj_weight = weights["self_attn_v_proj_weight"]
o_proj_weight = weights["o_proj_weight"]

# FFN weights
layernormFFN_weight = weights["layernormFFN_weight"]
up_proj_weight = weights["up_proj_weight"]
gate_proj_weight = weights["gate_proj_weight"]
down_proj_weight = weights["down_proj_weight"]

# Final layer normalization
model_layernorm_weight = weights["model_layernorm_weight"]

# Final vocabulary projection. Here, "head" is not the same as head in multi-head attention.
# "head" often refers to the final layer or component that maps the model's internal 
# hidden representations (embeddings) to the output space
lm_head_weight = weights["lm_head_weight"]

# 4- 
#######################################################
# Main Generation Loop: One Iteration/ Forward Pass
#######################################################
# The output is the next token id
def run_one_iteration(input_ids: list) -> int:
    # --- Multi-Headed Causal Self-Attention ---

    input_tensor = torch.tensor(input_ids, dtype=torch.int32, device='cuda')
    # print(f"Size of input tensor: {input_tensor.size()}") # (seq_len)

    # Create hidden state tensor by indexing into the embedding matrix with input tensor
    # For each token id, this will pluck the embedding vector for that token from the embedding matrix
    # and form a multi-dimensional tensor
    # I: (seq_len): (seq_len)
    # O: (seq_len, hidden_dim): (seq_len, 4096)
    hidden_state = embedding[input_tensor] 

    for current_layer in range(layers):
        # RMSNorm for each vector of user requests
        # I: (seq_len, hidden_dim): (seq_len, 4096)
        # O: (seq_len, hidden_dim): (seq_len, 4096)
        # Hidden dimension is also referred to as the embedding dimension or d_model.
        # This is your assignment I guess, but you can also use the torch functions for rms norm
        # This is along a row (-1: last dimension), meaning you iterate over each column (the second dimension). 
        rms = torch.sqrt(torch.mean(hidden_state ** 2, dim=-1, keepdim=True) + 1e-5) # (seq_len, 1)
        normalized_x = hidden_state / rms # (seq_len, hidden_dim): (seq_len, 4096)
        # To complete normalization, we have an elelment-wise multiplication of the 
        # normalized vectors with the layernorm weights
        # I_1 / x: (seq_len, hidden_dim): (seq_len, 4096) 
        # I_2 / w: (hidden_dim) -> (seq_len, hidden_dim): (seq_len, 4096) # broadcasting
        # 
        # O: (seq_len, hidden_dim): (seq_len, 4096)
        # print("Layernorm weight shape:", layernormAttn_weight[current_layer].shape)
        x = normalized_x * layernormAttn_weight[current_layer]
        # This goes through the rmsnorm, the first block in the diagram

        # QKV projection
        # I_1 / x: (seq_len, hidden_dim): (seq_len, 4096)
        # I_2 / w: (num_qo_heads * head_dim, hidden_dim): (32 * 128, 4096)
        # O: (seq_len, num_qo_heads * head_dim): (seq_len, 32 * 128)
        # How to know to take the transpose of the weight here?
        # Since I_2 is (num_qo_heads * head_dim, hidden_dim), we transpose it to (hidden_dim, num_qo_heads * head_dim)
        q = x.matmul(self_attn_q_proj_weight[current_layer].t()) # This means each q is sliced from the hidden state to create query heads
        # For k, v, num_kv_heads = 8, so w is (num_kv_heads * head_dim, hidden_dim): (8 * 128, 4096)
        k = x.matmul(self_attn_k_proj_weight[current_layer].t()) # O: (seq_len, num_kv_heads * head_dim): (seq_len, 8 * 128)
        v = x.matmul(self_attn_v_proj_weight[current_layer].t()) # O: (seq_len, num_kv_heads * head_dim): (seq_len, 8 * 128)

        # RoPE (Rotary Positional Embedding)
        apply_rope(q, output=q, head_dim=head_dim, offset=0)
        apply_rope(k, output=k, head_dim=head_dim, offset=0)

        # Compute sub-components of q, k, v for each head
        # I: (seq_len, num_qo_heads * head_dim): (seq_len, 32 * 128)
        # -1 allows torch to infer the first dimension (seq_len)
        sub_q = q.view(-1, num_qo_heads, head_dim) # (seq_len, num_qo_heads, head_dim): (seq_len, 32, 128)
        sub_k = k.view(-1, num_kv_heads, head_dim) # (seq_len, num_kv_heads, head_dim): (seq_len, 8, 128)
        sub_v = v.view(-1, num_kv_heads, head_dim) # (seq_len, num_kv_heads, head_dim): (seq_len, 8, 128)

        # Compute some attention-related values
        scale = 1.0 / (head_dim ** 0.5)
        group_size = num_qo_heads // num_kv_heads
        # The sequence length for q and k is needed to compute the causal mask [slide]
        # The below is needed to obtain the dimensions of the attention score matrix
        n_q = sub_q.shape[0] # seq_len
        n_k = sub_k.shape[0] # seq_len 

        # Replicate sub_k and sub_v for each group of query heads
        # The underlying KV values are shared, but each query will attend to them differently
        # I: (seq_len, num_kv_heads, head_dim): (seq_len, 8, 128)
        # O: (seq_len, num_qo_heads, head_dim): (seq_len, 32, 128) # num_qo_heads = num_kv_heads * group_size
        sub_k = sub_k.repeat_interleave(group_size, dim=1)
        sub_v = sub_v.repeat_interleave(group_size, dim=1)

        # Rearrange q and k so the shapes are (num_qo_heads, seq_len, head_dim):
        # This is because the computation of attention scores requires taking the dot product between 
        # each query vector and each key vector across the sequence dimension. Specifically, for 
        # multi-head attention, the operation must independently occur across each attention head.
        sub_q_t = sub_q.permute(1, 0, 2) # (num_qo_heads, seq_len, head_dim): (32, seq_len, 128)
        sub_k_t = sub_k.permute(1, 0, 2) # (num_qo_heads, seq_len, head_dim): (32, seq_len, 128)

        # Compute attention scores
        # I_1/q: (num_qo_heads, seq_len, head_dim): (32, seq_len, 128)
        # I_2/k: (num_qo_heads, seq_len, head_dim): (32, seq_len, 128)
        # O: (num_qo_heads, seq_len, seq_len): (32, seq_len, seq_len)
        # We take the transpose of the keys to align the dimensions for matrix multiplication
        scores = torch.matmul(sub_q_t, sub_k_t.transpose(-2, -1)) * scale

        # Compute causal mask
        # Create a lower-triangular matrix with values set to True:
        causal_mask = torch.tril(torch.ones(n_q, n_k, dtype=torch.bool, device=scores.device)) # (seq_len, seq_len)
        # Apply the mask, logically inverting it to get the upper part to be True, set those masked positions to -inf
        # Expand the mask along the first dimension (1, seq_len, seq_len) to match the shape of the scores
        # https://pytorch.org/docs/stable/generated/torch.Tensor.masked_fill_.html#torch.Tensor.masked_fill_
        scores = scores.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))

        # Apply softmax along the last dimension (for each query against all keys) to get attention weights
        # (num_qo_heads, seq_len, seq_len) corresponds to (heads, queries, keys)
        attn_weights = torch.softmax(scores, dim=-1) # (num_qo_heads, seq_len, seq_len): (32, seq_len, seq_len)

        # Compute attention output by multiplying weights with the values. sub_v has shape (seq_len, num_qo_heads, head_dim)
        # Transpose the sub_v tensor to get the shape (num_qo_heads, seq_len, head_dim)
        v_t = sub_v.permute(1, 0, 2) # (num_qo_heads, seq_len, head_dim): (32, seq_len, 128)
        # I_1/w: (num_qo_heads, seq_len, seq_len): (32, seq_len, seq_len)
        # I_2/v: (num_qo_heads, seq_len, head_dim): (32, seq_len, 128)
        # O: (num_qo_heads, seq_len, head_dim): (32, seq_len, 128)
        attn_output = torch.matmul(attn_weights, v_t)

        # Go back to single-head from multi-head attention. We combine the outputs from all heads
        attn_output = attn_output.permute(1, 0, 2) # (seq_len, num_qo_heads, head_dim): (seq_len, 32, 128)
        # Reshape to combine all heads' outputs into a single tensor
        # I: (seq_len, num_qo_heads, head_dim): (seq_len, 32, 128)
        # O: (seq_len, hidden_dim): (seq_len, 4096) # hidden_dim = num_qo_heads * head_dim
        attn_output = attn_output.reshape(-1, num_qo_heads * head_dim)

        # Output projection and residual connection. There is a residual connection in the attention block
        # So we add the original input to output projection. Seminal paper: https://arxiv.org/abs/1512.03385
        # I_1/attn: (seq_len, hidden_dim): (seq_len, 4096)
        # I_2/w: (hidden_dim, hidden_dim): (4096, 4096)
        # O: (seq_len, hidden_dim): (seq_len, 4096)
        o_proj_residual = attn_output.matmul(o_proj_weight[current_layer].t()) + hidden_state

        # --- Feed-Forward Network (FFN) ---

        # RMSNorm before FFN
        rms = torch.sqrt(torch.mean(o_proj_residual ** 2, dim=-1, keepdim=True) + 1e-5) # (seq_len, 1)
        normalized_x = o_proj_residual / rms # (seq_len, hidden_dim): (seq_len, 4096)
        layernormFFN_output = normalized_x.to(torch.float16) * layernormFFN_weight[current_layer] # (seq_len, hidden_dim): (seq_len, 4096)
        # Up projection
        # I_1/ln: (seq_len, hidden_dim): (seq_len, 4096)
        # I_2/w: (hidden_dim * 4,hidden_dim): (16384, 4096)
        # O: (seq_len, hidden_dim * 4): (seq_len, 16384)
        up_proj_output = layernormFFN_output.matmul(up_proj_weight[current_layer].t())
        # Gate
        # I_1/ln: (seq_len, hidden_dim): (seq_len, 4096)
        # I_2/w: (hidden_dim * 4,hidden_dim): (16384, 4096)
        # O: (seq_len, hidden_dim * 4): (seq_len, 16384)
        gate_proj_output = layernormFFN_output.matmul(gate_proj_weight[current_layer].t())
    
        # Gate + SiLU = SwiGLU
        # I_1/up: (seq_len, hidden_dim * 4): (seq_len, 16384)
        # I_2/gate: (seq_len, hidden_dim * 4): (seq_len, 16384)
        # O: (seq_len, hidden_dim * 4): (seq_len, 16384)
        activation_output = up_proj_output * torch.nn.functional.silu(gate_proj_output) # (seq_len, hidden_dim * 4): (seq_len, 16384)

        # Down projection 
        # I_1/act: (seq_len, hidden_dim * 4): (seq_len, 16384)
        # I_2/w: (hidden_dim, hidden_dim * 4): (4096, 16384)
        # O: (seq_len, hidden_dim): (seq_len, 4096)
        down_proj_output = activation_output.matmul(down_proj_weight[current_layer].t())
        # Residual connection
        hidden_state = down_proj_output + o_proj_residual

    # --- Final Layer Normalization, Projection to Vocabulary, Sampling ---
 
    # RMSNorm
    rms = torch.sqrt(torch.mean(hidden_state ** 2, dim=-1, keepdim=True) + 1e-5) # (seq_len, 1)
    normalized_x = hidden_state / rms # size (seq_len, hidden_dim): (seq_len, 4096)
    model_output = normalized_x.to(torch.float16) * model_layernorm_weight # size (seq_len, hidden_dim): (seq_len, 4096)

    # Project to vocabulary
    # I: (seq_len, hidden_dim): (seq_len, 4096)
    # O: (seq_len, vocab_size): (seq_len, 128256)
    # For each token in the sequence, this gives a probability distribution over the vocabulary
    logits = model_output.matmul(lm_head_weight.t())
    # Sum the numbers in the logits and print
    # Pick the next token with the highest probability
    sample_output = torch.argmax(logits, dim=1) # (seq_len, )
    # Return the next token following the last token in the input sequence
    # we need the -1 because sample_output contains the next tokens for all input tokens
    return sample_output[-1].item()

# 2- 
########################################
# Main Loop: Text Generation
########################################
def _demo_generation() -> None:
    input_string = "The University of Washington is"
    input_ids = tokenizer.encode(input_string)
    # 3- DEBUG
    # print("Size of input string", len(input_string.split()))
    # print("Size of input ids/Sequence Length:", len(input_ids))

    # Output will be appended at the end of input ids
    output_ids = input_ids.copy()
    # DEBUG
    # print(f"Token IDs: {input_ids}")

    # Generate token for rounds time. Each iteration goes through all layers
    iterations = 10
    for _round in range(iterations):
        new_token = run_one_iteration(output_ids)
        output_ids.append(new_token)

    # Skip special tokens like <start of string> or <end of string>
    output_string = tokenizer.decode(output_ids, skip_special_tokens=True)
    print(f"Output String: {output_string}")


if __name__ == "__main__":
    _demo_generation()
