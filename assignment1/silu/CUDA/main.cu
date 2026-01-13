#include <cuda_runtime.h>
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
    float atol = 1e-6f; // this seems needed to get things to match
    bool success = true;
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            int index = i * num_cols + j;
            float expected = silu(host_input[index]);
            if (fabs(host_output[index] - expected) > atol) {
                printf("Mismatch at (%d, %d): %f != %f\n", i, j, host_output[index], expected);
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