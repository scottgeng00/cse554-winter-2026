import torch
from silu_triton_kernel import silu_triton, silu_triton_2d

NUM_ITERS_FOR_BENCHMARK = 1000
MIN_NUM_MEM_ACCESSES_PER_ITER = 2
KERNEL_IMPLEMENTATION = silu_triton

# Test the Triton kernel
torch.manual_seed(0)
size = (8192, 8192)

x = torch.rand(size, device="cuda")

output_torch = torch.nn.SiLU()(x)
output_triton = KERNEL_IMPLEMENTATION(x)

print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')
assert torch.allclose(output_torch, output_triton), "mismatched silu result"


start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(NUM_ITERS_FOR_BENCHMARK):
    y = KERNEL_IMPLEMENTATION(x)
end.record()
torch.cuda.synchronize()


each_iter_time = start.elapsed_time(end) / 1000 / NUM_ITERS_FOR_BENCHMARK  # Convert milliseconds to seconds
print("Time taken for silu:", each_iter_time, "seconds")
print("Bandwidth: ", MIN_NUM_MEM_ACCESSES_PER_ITER * x.element_size() * x.numel() / each_iter_time / 1e9, "GB/s")