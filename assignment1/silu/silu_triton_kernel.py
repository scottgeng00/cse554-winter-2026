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


# lets test it out with a 2d grid size just to see how it perfs
BLOCK_ROW = 32
BLOCK_COL = 32

@triton.jit
def silu_kernel_2d(
    input_ptr,
    output_ptr,
    n_rows,
    n_cols,
    BLOCK_ROW: tl.constexpr,
    BLOCK_COL: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    block_start_m = pid_m * BLOCK_ROW
    block_start_n = pid_n * BLOCK_COL

    offsets_rows = block_start_m + tl.arange(0, BLOCK_ROW)
    offsets_cols = block_start_n + tl.arange(0, BLOCK_COL)
    
    offsets_flat = offsets_rows[:, None] * n_cols + offsets_cols[None, :]
    mask = (offsets_rows[:, None] < n_rows) & (offsets_cols[None, :] < n_cols)
    x = tl.load(input_ptr + offsets_flat, mask=mask)
    sigmoid_x = 1 / (1 + tl.exp(-x))
    output = x * sigmoid_x
    tl.store(output_ptr + offsets_flat, output, mask=mask)


def silu_triton_2d(input):
    output = torch.empty_like(input)
    n_rows, n_cols = input.shape

    grid = lambda meta: (triton.cdiv(n_rows, meta["BLOCK_ROW"]), triton.cdiv(n_cols, meta["BLOCK_COL"]))
    silu_kernel_2d[grid](input, output, n_rows, n_cols, BLOCK_ROW=BLOCK_ROW, BLOCK_COL=BLOCK_COL)
    return output