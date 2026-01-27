#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCKSIZE 256

// Section 2 Q2: we care about (8192, 8192) matrices.

// ============== ORIGINAL VERSION ==============
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

// Kernel-only version for benchmarking (data copied once before timing loop)
void rms_norm_matrix_kernel_only(float *d_input, float *d_weight, float *d_output, int rows, int cols, float epsilon) {
    dim3 num_blocks(rows);
    dim3 num_threads(BLOCKSIZE);
    rms_norm_matrix_kernel<<<num_blocks, num_threads>>>(d_input, d_weight, d_output, rows, cols, epsilon);
}

// ============== VECTORIZED VERSION (float4 loads) ==============
__global__ void rms_norm_matrix_kernel_vectorized(float *input, float *weight, float *output, int rows, int cols, float epsilon) {
    __shared__ float sdata[BLOCKSIZE];
    int row = blockIdx.x;
    int tid = threadIdx.x;

    // Pointer to start of this row
    float *row_input = input + row * cols;
    float *row_output = output + row * cols;

    // Phase 1: Compute sum of squares with float4 loads
    float partial_sum_sq = 0.0f;
    
    int cols4 = cols / 4;
    float4 *input4 = reinterpret_cast<float4*>(row_input);
    
    // Each thread processes multiple float4 chunks
    for (int i = tid; i < cols4; i += blockDim.x) {
        float4 val = input4[i];
        partial_sum_sq += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
    }
    
    // Handle remaining elements (if cols not divisible by 4)
    int remaining_start = cols4 * 4;
    for (int col = remaining_start + tid; col < cols; col += blockDim.x) {
        float val = row_input[col];
        partial_sum_sq += val * val;
    }

    sdata[tid] = partial_sum_sq;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    float rrms = rsqrtf((sdata[0] / cols) + epsilon);

    // Phase 2: Normalize with float4 loads/stores
    float4 *weight4 = reinterpret_cast<float4*>(weight);
    float4 *output4 = reinterpret_cast<float4*>(row_output);

    for (int i = tid; i < cols4; i += blockDim.x) {
        float4 in = input4[i];
        float4 w = weight4[i];
        float4 out;
        out.x = in.x * w.x * rrms;
        out.y = in.y * w.y * rrms;
        out.z = in.z * w.z * rrms;
        out.w = in.w * w.w * rrms;
        output4[i] = out;
    }

    // Handle remaining elements
    for (int col = remaining_start + tid; col < cols; col += blockDim.x) {
        row_output[col] = row_input[col] * weight[col] * rrms;
    }
}

// Kernel-only version with vectorized loading
void rms_norm_matrix_kernel_only_vectorized(float *d_input, float *d_weight, float *d_output, int rows, int cols, float epsilon) {
    dim3 num_blocks(rows);
    dim3 num_threads(BLOCKSIZE);
    rms_norm_matrix_kernel_vectorized<<<num_blocks, num_threads>>>(d_input, d_weight, d_output, rows, cols, epsilon);
}
