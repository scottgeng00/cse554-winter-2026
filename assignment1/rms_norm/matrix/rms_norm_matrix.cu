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

    rms_norm_matrix_kernel<<<num_blocks, num_threads>>>(device_input, device_weight, device_output, rows, cols, epsilon);

    // copy output back to host
    cudaMemcpy(output, device_output, size_matrix, cudaMemcpyDeviceToHost);
    cudaFree(device_weight);
    cudaFree(device_input);
    cudaFree(device_output);
}

void rms_norm_matrix_time(float *input, float *weight, int rows, int cols, float epsilon) {
    size_t size_matrix = (size_t)rows * cols * sizeof(float);
    size_t size_weight = cols * sizeof(float);

    float *device_input, *device_output, *device_weight;
    cudaMalloc((void**)&device_input, size_matrix);
    cudaMalloc((void**)&device_output, size_matrix);
    cudaMalloc((void**)&device_weight, size_weight);

    cudaMemcpy(device_input, input, size_matrix, cudaMemcpyHostToDevice);
    cudaMemcpy(device_weight, weight, size_weight, cudaMemcpyHostToDevice);

    dim3 num_blocks(rows);
    dim3 num_threads(BLOCKSIZE);

    // Warmup run
    rms_norm_matrix_kernel<<<num_blocks, num_threads>>>(device_input, device_weight, device_output, rows, cols, epsilon);
    cudaDeviceSynchronize();

    const int NUM_ITERS = 100;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < NUM_ITERS; i++) {
        rms_norm_matrix_kernel<<<num_blocks, num_threads>>>(device_input, device_weight, device_output, rows, cols, epsilon);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0;
    cudaEventElapsedTime(&total_ms, start, stop);
    float avg_ms = total_ms / NUM_ITERS;

    /*
     * minimum memory accesses for rms_norm is 2: one read, one write.
     * So total memory accesses is 2 * rows * cols.
     * this is assuming some theoretically perfect kernel.
     */
    double total_bytes = (double)rows * cols * 2.0 * sizeof(float);
    double bandwidth_GBs = (total_bytes / 1e9) / (avg_ms / 1e3);
    double peak_bandwidth = 672.0;
    double utilization = (bandwidth_GBs / peak_bandwidth) * 100.0;

    printf("Time: %.4f ms (avg over %d iters)\n", avg_ms, NUM_ITERS);
    printf("Theoretical memory accesses: %.2f MB\n", total_bytes / 1e6);
    printf("Bandwidth: %.2f GB/s (%.1f%% of %.0f GB/s peak)\n", bandwidth_GBs, utilization, peak_bandwidth);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(device_weight);
    cudaFree(device_input);
    cudaFree(device_output);
}
