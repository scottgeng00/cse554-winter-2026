#include<cuda_runtime.h>
#include "rms_norm_vector.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define atol 1e-3f
#define NUM_ITERS 100

float random_float() {
    return (static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) * 2.0f - 1.0f;
}

void rms_norm_cpu(float *input, float *weight, float *output, int rows, int cols, float epsilon) {
    for (int row = 0; row < rows; row++) {
        float sum_sq = 0.0f;
        for (int col = 0; col < cols; col++) {
            sum_sq += input[row * cols + col] * input[row * cols + col];
        }
        float rms = sqrtf((sum_sq / (float) cols) + epsilon);
        for (int col = 0; col < cols; col++) {
            output[row * cols + col] = input[row * cols + col] / rms * weight[col];
        }
    }
}

int main() {
    int rows = 1;
    int cols = 1024 * 1024;
    float epsilon = 1e-3f;
    size_t matrix_size = (size_t)rows * cols;

    float *h_input, *h_weight, *h_output_gpu, *h_output_cpu;
    h_input = (float*)malloc(matrix_size * sizeof(float));
    h_weight = (float*)malloc(cols * sizeof(float));
    h_output_gpu = (float*)malloc(matrix_size * sizeof(float));
    h_output_cpu = (float*)malloc(matrix_size * sizeof(float));

    // Initialize input data
    srand(42);
    for (size_t i = 0; i < matrix_size; i++) {
        h_input[i] = random_float();
    }
    for (int i = 0; i < cols; i++) {
        h_weight[i] = random_float() + 1.0f;
    }

    // Run CPU reference
    rms_norm_cpu(h_input, h_weight, h_output_cpu, rows, cols, epsilon);

    // ========== Check Version 1: Original ==========
    printf("=== Correctness Check ===\n");
    rms_norm_vector(h_input, h_weight, h_output_gpu, cols, epsilon);

    int errors = 0;
    for (size_t i = 0; i < matrix_size && errors < 5; i++) {
        float diff = fabsf(h_output_gpu[i] - h_output_cpu[i]);
        if (diff > atol) {
            printf("V1 Mismatch at %zu: GPU=%.6f, CPU=%.6f\n", i, h_output_gpu[i], h_output_cpu[i]);
            errors++;
        }
    }

    if (errors == 0) {
        printf("Version 1 (Original): CORRECT\n");
    } else {
        printf("Version 1 (Original): INCORRECT\n");
    }

    // ========== Check Version 2: Vectorized ==========
    float *d_input_check, *d_weight_check, *d_output_check, *d_sqsum_check;
    cudaMalloc((void**)&d_input_check, cols * sizeof(float));
    cudaMalloc((void**)&d_weight_check, cols * sizeof(float));
    cudaMalloc((void**)&d_output_check, cols * sizeof(float));
    cudaMalloc((void**)&d_sqsum_check, sizeof(float));

    cudaMemcpy(d_input_check, h_input, cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight_check, h_weight, cols * sizeof(float), cudaMemcpyHostToDevice);

    rms_norm_vector_kernel_only_vectorized(d_input_check, d_weight_check, d_output_check, 
                                            d_sqsum_check, cols, epsilon);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output_gpu, d_output_check, cols * sizeof(float), cudaMemcpyDeviceToHost);

    errors = 0;
    for (size_t i = 0; i < matrix_size && errors < 5; i++) {
        float diff = fabsf(h_output_gpu[i] - h_output_cpu[i]);
        if (diff > atol) {
            printf("V2 Mismatch at %zu: GPU=%.6f, CPU=%.6f\n", i, h_output_gpu[i], h_output_cpu[i]);
            errors++;
        }
    }

    if (errors == 0) {
        printf("Version 2 (Vectorized): CORRECT\n");
    } else {
        printf("Version 2 (Vectorized): INCORRECT\n");
    }

    cudaFree(d_input_check);
    cudaFree(d_weight_check);
    cudaFree(d_output_check);
    cudaFree(d_sqsum_check);

    // ========== Benchmark (kernel only, no memcpy) ==========
    printf("\n=== RMS Norm Vector Benchmark ===\n");
    printf("Vector size: %d\n", cols);
    printf("Iterations: %d\n", NUM_ITERS);

    // Allocate GPU memory once
    float *d_input, *d_weight, *d_output, *d_sqsum;
    cudaMalloc((void**)&d_input, cols * sizeof(float));
    cudaMalloc((void**)&d_weight, cols * sizeof(float));
    cudaMalloc((void**)&d_output, cols * sizeof(float));
    cudaMalloc((void**)&d_sqsum, sizeof(float));

    cudaMemcpy(d_input, h_input, cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight, cols * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    float avg_time_ms, avg_time_s;
    size_t min_bytes = 2 * cols * sizeof(float);
    float min_throughput_gb_s;

    // ========== Version 1: Original ==========
    printf("\n--- Version 1: Original ---\n");

    rms_norm_vector_kernel_only(d_input, d_weight, d_output, d_sqsum, cols, epsilon);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < NUM_ITERS; i++) {
        rms_norm_vector_kernel_only(d_input, d_weight, d_output, d_sqsum, cols, epsilon);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop);
    avg_time_ms = milliseconds / NUM_ITERS;
    avg_time_s = avg_time_ms / 1000.0f;
    min_throughput_gb_s = (min_bytes / 1e9) / avg_time_s;

    printf("Average kernel time: %.4f ms\n", avg_time_ms);
    printf("Memory throughput: %.2f GB/s\n", min_throughput_gb_s);

    // ========== Version 2: Vectorized ==========
    printf("\n--- Version 2: Vectorized (float4) ---\n");

    rms_norm_vector_kernel_only_vectorized(d_input, d_weight, d_output, d_sqsum, cols, epsilon);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < NUM_ITERS; i++) {
        rms_norm_vector_kernel_only_vectorized(d_input, d_weight, d_output, d_sqsum, cols, epsilon);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop);
    avg_time_ms = milliseconds / NUM_ITERS;
    avg_time_s = avg_time_ms / 1000.0f;
    min_throughput_gb_s = (min_bytes / 1e9) / avg_time_s;

    printf("Average kernel time: %.4f ms\n", avg_time_ms);
    printf("Memory throughput: %.2f GB/s\n", min_throughput_gb_s);

    // ========== Summary ==========
    printf("\n--- Summary ---\n");
    printf("Minimum memory accesses: 2 per element (1 read + 1 write)\n");
    printf("Minimum memory traffic: %.2f MB\n", min_bytes / 1e6);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
    cudaFree(d_sqsum);

    free(h_input);
    free(h_weight);
    free(h_output_gpu);
    free(h_output_cpu);
    return 0;
}
