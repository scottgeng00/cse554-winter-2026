#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCKSIZE 256

// Section 2 Q2: we care about (8192, 8192) matrices.

// our kernel operates on rows, with one block per row.
// we take a reduction approach, fairly similar to reduction3.cu
__global__ void rms_norm_matrix_kernel(float *input, float *weight, float *output, int rows, int cols, float epsilon) {
    // use static shared memory to store sumsq
    __shared__ float sdata[BLOCKSIZE];
    int row = blockIdx.x; 
    int tid = threadIdx.x;

    float partial_sum_sq = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        // load input value
        float val = input[row * cols + col];
        partial_sum_sq += val * val;
    }
    sdata[tid] = partial_sum_sq;
    __syncthreads();
    
    // doing a reduction, like reduction3.cu
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // rsqrtf is faster than 1.0f / sqrtf
    // after this we have the reciprocal of the rms, so we can just multiply by the weight and input.
    float rrms = rsqrtf((sdata[0] / cols) + epsilon);

    // finally, just do the division everywehere.
    for (int col = tid; col < cols; col += blockDim.x) {
        output[row * cols + col] = input[row * cols + col] * weight[col] * rrms;
    }
}

void rms_norm_matrix(float *input, float *weight, float *output, int rows, int cols, float epsilon) {
    // two sizes, since weight is col-vector
    size_t size_matrix = rows * cols * sizeof(float);
    size_t size_weight = cols * sizeof(float);

    // allocate gpu memory
    float *device_input, *device_output, *device_weight;
    cudaMalloc((void**)&device_input, size_matrix);
    cudaMalloc((void**)&device_output, size_matrix);
    cudaMalloc((void**)&device_weight, size_weight);

    // copy weight and input matrix to gpu
    cudaMemcpy(device_input, input, size_matrix, cudaMemcpyHostToDevice);
    cudaMemcpy(device_weight, weight, size_weight, cudaMemcpyHostToDevice);

    dim3 num_blocks(rows);      // One block per row
    dim3 num_threads(BLOCKSIZE); // Threads cooperate within each row

    // Time the kernel
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    rms_norm_matrix_kernel<<<num_blocks, num_threads>>>(device_input, device_weight, device_output, rows, cols, epsilon);
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float ms;
    cudaEventElapsedTime(&ms, start, end);

    // Actual bandwidth: read input (2x) + read weight + write output
    size_t actual_bytes = 3 * size_matrix + size_weight;
    float achieved_bandwidth_gb = (actual_bytes / 1e9) / (ms / 1000.0f);
    // Theoretical bandwidth calculation
    size_t effective_bytes = 2 * size_matrix + size_weight; // assuming input read once, output written once, weight read once
    float effective_bandwidth_gb = (effective_bytes / 1e9) / (ms / 1000.0f);
    printf("Kernel time: %.3f ms, Actual Bandwidth: %.1f GB/s, Algo Bandwidth: %.1f GB/s\n", ms, achieved_bandwidth_gb, effective_bandwidth_gb);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    // copy output back to host
    cudaMemcpy(output, device_output, size_matrix, cudaMemcpyDeviceToHost);
    cudaFree(device_weight);
    cudaFree(device_input);
    cudaFree(device_output);
}
