#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

struct GemmShape {
    int N;
    int K;
};

float profile_gemm(cublasHandle_t handle, int M, int N, int K) {

    size_t size_A = (size_t)M * K;
    size_t size_B = (size_t)K * N;
    size_t size_C = (size_t)M * N;

    half *host_A = new half[size_A];
    half *host_B = new half[size_B];
    half *host_C = new half[size_C];

    // initing host arrays
    for (size_t i = 0; i < size_A; i++) host_A[i] = (half)i;
    for (size_t i = 0; i < size_B; i++) host_B[i] = (half)i;

    half *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size_A * sizeof(half));
    cudaMalloc((void**)&d_B, size_B * sizeof(half));
    cudaMalloc((void**)&d_C, size_C * sizeof(half));

    cudaMemcpy(d_A, host_A, size_A * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, host_B, size_B * sizeof(half), cudaMemcpyHostToDevice);

    // alpha, beta for scaling the result.
    // we set to defaults for now.
    half alpha_h = __float2half(1.0f);
    half beta_h  = __float2half(0.0f);

    // Warmup
    int warm_up_count = 100;
    int profile_count = 100;
    size_t L2_size = 50 * 1024 * 1024;
    for (int i = 0; i < warm_up_count; ++i) {
        cublasGemmEx(handle,
                     CUBLAS_OP_N, CUBLAS_OP_N,
                     N, M, K,
                     &alpha_h,
                     d_B, CUDA_R_16F, N,
                     d_A, CUDA_R_16F, K,
                     &beta_h,
                     d_C, CUDA_R_16F, N,
                     CUBLAS_COMPUTE_16F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error after warmup (M=" << M << " N=" << N
                  << " K=" << K << "): " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    int* clear_l2_buffer;
    cudaMalloc(&clear_l2_buffer, L2_size);

    float total_ms = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < profile_count; ++i) {
        cudaMemset(clear_l2_buffer, 0, L2_size);
        cudaEventRecord(start);
        cublasGemmEx(handle,
                     CUBLAS_OP_N, CUBLAS_OP_N,
                     N, M, K,
                     &alpha_h,
                     d_B, CUDA_R_16F, N,
                     d_A, CUDA_R_16F, K,
                     &beta_h,
                     d_C, CUDA_R_16F, N,
                     CUBLAS_COMPUTE_16F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }

    float average_time = total_ms / profile_count;

    cudaFree(clear_l2_buffer);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] host_A;
    delete[] host_B;
    delete[] host_C;

    return average_time;
}

int main() {
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    std::vector<GemmShape> shapes = {
        {512, 512}, {4096, 4096}, {14336, 4096},  {4096, 1024}, {1024, 4096}
    };

    std::vector<int> M_values = {128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048};

    std::ofstream csv("Section1/gemm_perf.csv");
    csv << "batch_size,N,K,library,tflops\n";

    for (auto& shape : shapes) {
        for (int M : M_values) {
            float avg_ms = profile_gemm(handle, M, shape.N, shape.K);

            // error, just return -1
            if (avg_ms <= 0.0f) {
                std::cerr << "Error in profile_gemm\n";
                return -1;
            }

            // flops for gemm is 2 * M * N * K
            double flops = 2.0 * M * shape.N * shape.K;
            // tflops adjust for ms and tera-bit
            double tflops = flops / (avg_ms * 1e-3) / 1e12;

            csv << M << "," << shape.N << "," << shape.K
                << ",cublas," << tflops << "\n";

            std::cout << "M=" << M << " N=" << shape.N << " K=" << shape.K
                      << "  avg_time=" << avg_ms << " ms  TFLOPS=" << tflops << "\n";
        }
    }

    csv.close();
    cublasDestroy(handle);
    return 0;
}
