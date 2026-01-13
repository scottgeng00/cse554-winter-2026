import torch
import triton
import triton.language as tl


BLOCK_SIZE = 128  # You can tune this parameter

@triton.jit
def silu_kernel(
    input_ptr,
    output_ptr,
    n_elements, # we can just treat the matrix as a flat array
    BLOCK_SIZE: tl.constexpr,
):
    # some bookkeeping thats identical to addition_triton.py
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # load input, compute result, store output
    x = tl.load(input_ptr + offsets, mask=mask)
    sigmoid_x = 1 / (1 + tl.exp(-x))
    output = x * sigmoid_x
    tl.store(output_ptr + offsets, output, mask=mask)


def silu_triton(input):
    output = torch.empty_like(input)
    n_elements = input.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    silu_kernel[grid](input, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output