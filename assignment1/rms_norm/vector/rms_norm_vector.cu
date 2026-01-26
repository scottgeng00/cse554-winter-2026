#include <cuda_runtime.h>
#include <stdio.h>
#include <cooperative_groups.h>

#define BLOCKSIZE 256
#define ELEMENTS_PER_BLOCK 8192

// defines how many local reductions to do when loading to shared memory
constexpr int read_iter = ELEMENTS_PER_BLOCK / BLOCKSIZE;

__global__ void sqsumKernel(float *d_input, float *d_output, int N) {
    __shared__ float sdata[ELEMENTS_PER_BLOCK];
    int tid = threadIdx.x;

    int i = blockIdx.x * blockDim.x * read_iter + threadIdx.x;
    float local_sum = 0;
    for (int j = 0; j < read_iter; j++) {
        if (i < N) {
            local_sum += d_input[i] * d_input[i];
        }
        i += blockDim.x;
    }
    sdata[tid] = local_sum;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    // Write result for this block to global memory
    if (tid == 0) {
        atomicAdd(d_output, sdata[0]);
    }
}

__global__ void rms_norm_vector_kernel(float *d_input, float *d_weight, float *d_output, float *sqsum, int cols, float epsilon) {
    float rms = sqrtf((*sqsum / cols) + epsilon);
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < cols) {
        d_output[index] = d_input[index] / rms * d_weight[index];
    }   
}


void rms_norm_vector(float *input, float *weight, float *output, int cols, float epsilon) {
    // two sizes, since weight is col-vector
    size_t size = cols * sizeof(float);

    // allocate gpu memory
    float *device_input, *device_output, *device_weight;
    cudaMalloc((void**)&device_input, size);
    cudaMalloc((void**)&device_output, size);
    cudaMalloc((void**)&device_weight, size);

    // copy weight and input matrix to gpu
    cudaMemcpy(device_input, input, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_weight, weight, size, cudaMemcpyHostToDevice);

    // let's try to achieve this via two kernel launches: first reduce, then normalize.
    float *d_sqsum;
    cudaMalloc((void**)&d_sqsum, sizeof(float));
    cudaMemset(d_sqsum, 0, sizeof(float));

    dim3 num_blocks_reduce((cols + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK); // reduce with this blocksize
    dim3 num_blocks_normalize((cols + BLOCKSIZE - 1) / BLOCKSIZE);
    dim3 num_threads(BLOCKSIZE); 

    // Time the kernel
    cudaEvent_t start_reduce, end_reduce, start_normalize, end_normalize;
    cudaEventCreate(&start_reduce);
    cudaEventCreate(&end_reduce);
    cudaEventCreate(&start_normalize);
    cudaEventCreate(&end_normalize);

    // maybe warmup reduction
    // sqsumKernel<<<num_blocks_reduce, num_threads>>>(device_input, d_sqsum, cols);
    // cudaDeviceSynchronize();
    // cudaMemset(d_sqsum, 0, sizeof(float));

    // time the reduction
    cudaEventRecord(start_reduce);
    sqsumKernel<<<num_blocks_reduce, num_threads>>>(device_input, d_sqsum, cols);
    cudaEventRecord(end_reduce);
    cudaEventSynchronize(end_reduce);

    // time the normalization
    cudaEventRecord(start_normalize);
    rms_norm_vector_kernel<<<num_blocks_normalize, num_threads>>>(device_input, device_weight, device_output, d_sqsum, cols, epsilon);
    cudaEventRecord(end_normalize);
    cudaEventSynchronize(end_normalize);

    // calculate elapsed time
    float reduce_ms, normalize_ms;
    cudaEventElapsedTime(&reduce_ms, start_reduce, end_reduce);
    cudaEventElapsedTime(&normalize_ms, start_normalize, end_normalize);
    float total_ms = reduce_ms + normalize_ms;

    // Bandwidth: read input (2x) + read weight + write output
    // Actual bandwidth: read input (2x) + read weight + write output
    size_t actual_bytes = 4 * size;
    float achieved_bandwidth_gb = (actual_bytes / 1e9) / (total_ms / 1000.0f);
    size_t effective_bytes = 2 * size; // assuming input read once, output written once, weight read once
    float effective_bandwidth_gb = (effective_bytes / 1e9) / (total_ms / 1000.0f);

    printf("Kernel time: %.3f ms, Actual Bandwidth: %.1f GB/s, Algo Bandwidth: %.1f GB/s\n", total_ms, achieved_bandwidth_gb, effective_bandwidth_gb);
    printf("Reduction time: %.3f ms, Normalization time: %.3f ms, Total time: %.3f ms\n", reduce_ms, normalize_ms, total_ms);
    printf("Reduction bandwidth: %.1f GB/s\n", (size / 1e9) / (reduce_ms / 1000.0f));
    printf("Normalization bandwidth: %.1f GB/s\n", (3 * size / 1e9) / (normalize_ms / 1000.0f));

    cudaEventDestroy(start_reduce);
    cudaEventDestroy(end_reduce);
    cudaEventDestroy(start_normalize);
    cudaEventDestroy(end_normalize);

    // copy output back to host
    cudaMemcpy(output, device_output, size, cudaMemcpyDeviceToHost);
    cudaFree(d_sqsum);
    cudaFree(device_weight);
    cudaFree(device_input);
    cudaFree(device_output);
}
