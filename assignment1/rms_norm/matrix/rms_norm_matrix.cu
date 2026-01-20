#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCKSIZE 1024


__global__ void rms_norm_matrix_kernel(float *input, float *weight, float *output, int rows, int cols, float epsilon) {
    // rms works row-wise, across feature dimension.
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float sum_sq = 0;
        for (int col = 0; col < cols; col++) {
            sum_sq += input[row * cols + col] * input[row * cols + col];
        }
        float rms = sqrtf(sum_sq / cols); // cols = length of row.
        for (int col = 0; col < cols; col++) {
            output[row * cols + col] = input[row * cols + col] / (rms + epsilon) * weight[col];
        }
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

    dim3 num_blocks((rows + BLOCKSIZE - 1) / BLOCKSIZE);
    dim3 num_threads(BLOCKSIZE);

    rms_norm_matrix_kernel<<<num_blocks, num_threads>>>(device_input, device_weight, device_output, rows, cols, epsilon);

    // copy output back to host
    cudaMemcpy(output, device_output, size_matrix, cudaMemcpyDeviceToHost);
    cudaFree(device_weight);
    cudaFree(device_input);
    cudaFree(device_output);
}
