#include <cuda_runtime.h>
#include <cmath> // Required for expf
#include <stdio.h>
#include "silu.h"

float random_float() {
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

float silu(float x) {
    return x / (1.0f + expf(-x));
}

int main() {
    // Launch the kernel
    int num_rows = 8192;
    int num_cols = 8192;
    size_t size = num_rows * num_cols * sizeof(float);

    float *host_input, *host_output;

    host_input = (float*)malloc(size);
    host_output = (float*)malloc(size);

    for (int i = 0; i < num_rows * num_cols; i++) {
        host_input[i] = random_float();
    }

    silu(host_input, host_output, num_rows * num_cols);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Verify the result
    bool success = true;
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            float expected = silu(host_input[i * num_rows + j]);
            if (host_output[i * num_rows + j] != expected) {
                printf("Mismatch at (%d, %d): %f != %f\n", i, j, host_output[j * num_rows + i], expected);
                success = false;
                break;
            }
        }
    }
    if (success) printf("Success! All values match.\n");

    free(host_input);
    free(host_output);
    return 0;
}