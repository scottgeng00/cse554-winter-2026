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
