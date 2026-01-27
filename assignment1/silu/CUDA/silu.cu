#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCKSIZE 1024

__global__ void silu_kernel(float *input, float *output, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        float x = input[index];
        output[index] = x / (1.0f + expf(-x));
    }
}

void silu(float *input, float *output, int n) {
    size_t size = n * sizeof(float);

    float *device_input, *device_output;
    cudaMalloc((void**)&device_input, size);
    cudaMalloc((void**)&device_output, size);

    cudaMemcpy(device_input, input, size, cudaMemcpyHostToDevice);

    dim3 num_blocks((n + BLOCKSIZE - 1) / BLOCKSIZE);
    dim3 num_threads(BLOCKSIZE);

    silu_kernel<<<num_blocks, num_threads>>>(device_input, device_output, n);

    // Copy output matrix back to host
    cudaMemcpy(output, device_output, size, cudaMemcpyDeviceToHost);
    cudaFree(device_input);
    cudaFree(device_output);
}

void silu_time(float *input, int n) {
    size_t size = n * sizeof(float);

    // Allocate device memory
    float *device_input, *device_output;
    cudaMalloc((void**)&device_input, size);
    cudaMalloc((void**)&device_output, size);

    cudaMemcpy(device_input, input, size, cudaMemcpyHostToDevice);

    dim3 num_blocks((n + BLOCKSIZE - 1) / BLOCKSIZE);
    dim3 num_threads(BLOCKSIZE);

    // Warmup run
    silu_kernel<<<num_blocks, num_threads>>>(device_input, device_output, n);
    cudaDeviceSynchronize();

    // Timed runs
    const int NUM_ITERS = 100;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < NUM_ITERS; i++) {
        silu_kernel<<<num_blocks, num_threads>>>(device_input, device_output, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0;
    cudaEventElapsedTime(&total_ms, start, stop);
    float avg_ms = total_ms / NUM_ITERS;

    /*
     * minimum memory accesses for silu is 2: one read one write.
     * So total memory accesses is 2 * n.
     * Then, we multiply by the size of the float to get total bytes moved.
     */
    double total_bytes = (double)n * 2.0 * sizeof(float);
    double bandwidth_GBs = (total_bytes / 1e9) / (avg_ms / 1e3);
    double peak_bandwidth = 672.0;  // from Quadro RTX 6000 info sheet
    double utilization = (bandwidth_GBs / peak_bandwidth) * 100.0;

    printf("Time: %.4f ms (avg over %d iters)\n", avg_ms, NUM_ITERS);
    printf("Theoretical memory accesses: %.2f MB\n", total_bytes / 1e6);
    printf("Bandwidth: %.2f GB/s (%.1f%% of %.0f GB/s peak)\n", bandwidth_GBs, utilization, peak_bandwidth);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(device_input);
    cudaFree(device_output);
}
