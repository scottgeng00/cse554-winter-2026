#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCKSIZE 256
#define ELEMENTS_PER_BLOCK 8192

// defines how many local reductions to do when loading to shared memory
constexpr int read_iter = ELEMENTS_PER_BLOCK / BLOCKSIZE;

// ============== ORIGINAL VERSION ==============
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

// Kernel-only version for benchmarking (data already on GPU) - ORIGINAL
void rms_norm_vector_kernel_only(float *d_input, float *d_weight, float *d_output, float *d_sqsum, int cols, float epsilon) {
    // Reset sqsum to 0 before each call
    cudaMemset(d_sqsum, 0, sizeof(float));

    dim3 num_blocks_reduce((cols + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK);
    dim3 num_blocks_normalize((cols + BLOCKSIZE - 1) / BLOCKSIZE);
    dim3 num_threads(BLOCKSIZE);

    sqsumKernel<<<num_blocks_reduce, num_threads>>>(d_input, d_sqsum, cols);
    rms_norm_vector_kernel<<<num_blocks_normalize, num_threads>>>(d_input, d_weight, d_output, d_sqsum, cols, epsilon);
}

// ============== VECTORIZED LOADING VERSION (float4 loads) ==============
#define NUM_BLOCKS_VECTORIZED 128

// Sqsum with float4 loads
__global__ void sqsumKernel_vectorized(float *d_input, float *d_output, int N) {
    __shared__ float sdata[BLOCKSIZE];
    
    float local_sum = 0.0f;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Vector loads (4 floats at a time)
    int N4 = N / 4;
    float4 *input4 = reinterpret_cast<float4*>(d_input);
    
    for (int i = tid; i < N4; i += stride) {
        float4 val = input4[i];
        local_sum += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
    }
    
    // Handle remaining elements
    int remaining_start = N4 * 4;
    for (int i = remaining_start + tid; i < N; i += stride) {
        float val = d_input[i];
        local_sum += val * val;
    }
    
    // Shared memory reduction within block
    sdata[threadIdx.x] = local_sum;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    // Use atomicAdd to accumulate block result
    if (threadIdx.x == 0) {
        atomicAdd(d_output, sdata[0]);
    }
}

// Normalize with float4 loads/stores
__global__ void rms_norm_kernel_vectorized(float *d_input, float *d_weight, float *d_output, 
                                            float *sqsum, int cols, float epsilon) {
    float rrms = rsqrtf((*sqsum / cols) + epsilon);  // Use rsqrt for efficiency
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Vector operations (4 floats at a time)
    int cols4 = cols / 4;
    float4 *input4 = reinterpret_cast<float4*>(d_input);
    float4 *weight4 = reinterpret_cast<float4*>(d_weight);
    float4 *output4 = reinterpret_cast<float4*>(d_output);
    
    for (int i = tid; i < cols4; i += stride) {
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
    int remaining_start = cols4 * 4;
    for (int i = remaining_start + tid; i < cols; i += stride) {
        d_output[i] = d_input[i] * d_weight[i] * rrms;
    }
}

// Kernel-only version with vectorized loading (2 kernels)
void rms_norm_vector_kernel_only_vectorized(float *d_input, float *d_weight, float *d_output, 
                                             float *d_sqsum, int cols, float epsilon) {
    // Reset sqsum to 0 before each call
    cudaMemset(d_sqsum, 0, sizeof(float));

    int num_blocks = NUM_BLOCKS_VECTORIZED;
    dim3 num_threads(BLOCKSIZE);

    sqsumKernel_vectorized<<<num_blocks, num_threads>>>(d_input, d_sqsum, cols);
    rms_norm_kernel_vectorized<<<num_blocks, num_threads>>>(d_input, d_weight, d_output, d_sqsum, cols, epsilon);
}
