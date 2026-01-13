import torch
from torch.profiler import profile, record_function, ProfilerActivity

def silu_torch(x: torch.Tensor) -> torch.Tensor:
    sigmoid_x = 1 / (1 + torch.exp(-x))
    return x * sigmoid_x

if __name__ == "__main__":
    NUM_ITERS_FOR_BENCHMARK = 100
    MIN_NUM_MEM_ACCESSES_PER_ITER = 5

    silu_torch_ref = torch.nn.SiLU()

    # make input
    tensor_shape = (8192, 8192)
    x = torch.rand(tensor_shape, device='cuda')

    # compute reference silu and check for correctness
    y_ref = silu_torch_ref(x)
    y = silu_torch(x)
    assert torch.allclose(y, y_ref), "mismatched silu result"

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(NUM_ITERS_FOR_BENCHMARK):
        y = silu_torch(x)
    end.record()
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        with_stack=True,
        record_shapes=True,
    ) as prof:
        # Use record_function to mark the section of code to be profiled
        with record_function("silu_torch"):
            for _ in range(10):
                y = silu_torch(x)

    prof.export_chrome_trace("silu_torch_trace.json")

    each_iter_time = start.elapsed_time(end) / 1000 / NUM_ITERS_FOR_BENCHMARK  # Convert milliseconds to seconds
    print("Time taken for silu:", each_iter_time, "seconds")
    print("Bandwidth: ", MIN_NUM_MEM_ACCESSES_PER_ITER * x.element_size() * x.numel() / each_iter_time / 1e9, "GB/s")