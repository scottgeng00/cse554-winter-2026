#include<cuda_runtime.h>
#include "rms_norm_vector.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define atol 1e-3f

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

    rms_norm_vector(h_input, h_weight, h_output_gpu, cols, epsilon);

    // Run CPU reference
    rms_norm_cpu(h_input, h_weight, h_output_cpu, rows, cols, epsilon);

    // Check result
    int errors = 0;
    for (size_t i = 0; i < matrix_size && errors < 5; i++) {
        float diff = fabsf(h_output_gpu[i] - h_output_cpu[i]);
        if (diff > atol) {
            printf("Mismatch at %zu: GPU=%.6f, CPU=%.6f\n", i, h_output_gpu[i], h_output_cpu[i]);
            errors++;
        }
    }

    if (errors == 0) {
        printf("Result is correct!\n");
    } else {
        printf("Result is incorrect!\n");
    }

    // Run timing
    rms_norm_vector_time(h_input, h_weight, cols, epsilon);

    // Free memory
    free(h_input);
    free(h_weight);
    free(h_output_gpu);
    free(h_output_cpu);
    return 0;
}
