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

    sqsumKernel<<<num_blocks_reduce, num_threads>>>(device_input, d_sqsum, cols);
    rms_norm_vector_kernel<<<num_blocks_normalize, num_threads>>>(device_input, device_weight, device_output, d_sqsum, cols, epsilon);

    // copy output back to host
    cudaMemcpy(output, device_output, size, cudaMemcpyDeviceToHost);
    cudaFree(d_sqsum);
    cudaFree(device_weight);
    cudaFree(device_input);
    cudaFree(device_output);
}

void rms_norm_vector_time(float *input, float *weight, int cols, float epsilon) {
    size_t size = cols * sizeof(float);

    // Allocate device memory
    float *device_input, *device_output, *device_weight, *d_sqsum;
    cudaMalloc((void**)&device_input, size);
    cudaMalloc((void**)&device_output, size);
    cudaMalloc((void**)&device_weight, size);
    cudaMalloc((void**)&d_sqsum, sizeof(float));

    cudaMemcpy(device_input, input, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_weight, weight, size, cudaMemcpyHostToDevice);

    dim3 num_blocks_reduce((cols + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK);
    dim3 num_blocks_normalize((cols + BLOCKSIZE - 1) / BLOCKSIZE);
    dim3 num_threads(BLOCKSIZE);

    // Warmup run
    cudaMemset(d_sqsum, 0, sizeof(float));
    sqsumKernel<<<num_blocks_reduce, num_threads>>>(device_input, d_sqsum, cols);
    rms_norm_vector_kernel<<<num_blocks_normalize, num_threads>>>(device_input, device_weight, device_output, d_sqsum, cols, epsilon);
    cudaDeviceSynchronize();

    // Timed runs
    const int NUM_ITERS = 100;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < NUM_ITERS; i++) {
        cudaMemset(d_sqsum, 0, sizeof(float));
        sqsumKernel<<<num_blocks_reduce, num_threads>>>(device_input, d_sqsum, cols);
        rms_norm_vector_kernel<<<num_blocks_normalize, num_threads>>>(device_input, device_weight, device_output, d_sqsum, cols, epsilon);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0;
    cudaEventElapsedTime(&total_ms, start, stop);
    float avg_ms = total_ms / NUM_ITERS;

    /*
     * minimum memory accesses for rms_norm is 2: one read, one write.
     * So total memory accesses is 2 * cols (rows == 1 for this problem).
     */
    double total_bytes = (double)cols * 2.0 * sizeof(float);
    double bandwidth_GBs = (total_bytes / 1e9) / (avg_ms / 1e3);
    double peak_bandwidth = 672.0;  // from Quadro RTX 6000 info sheet
    double utilization = (bandwidth_GBs / peak_bandwidth) * 100.0;

    printf("Time: %.4f ms (avg over %d iters)\n", avg_ms, NUM_ITERS);
    printf("Theoretical memory accesses: %.2f MB\n", total_bytes / 1e6);
    printf("Bandwidth: %.2f GB/s (%.1f%% of %.0f GB/s peak)\n", bandwidth_GBs, utilization, peak_bandwidth);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_sqsum);
    cudaFree(device_weight);
    cudaFree(device_input);
    cudaFree(device_output);
}
