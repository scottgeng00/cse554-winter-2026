#include "copy_first_column.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

const int ROWS = 8192;
const int COLS = 65536;

int main() {
    // Use regular malloc (pageable memory)
    float *h_matrix = (float*)malloc((size_t)ROWS * COLS * sizeof(float));
    
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            h_matrix[(size_t)i * COLS + j] = static_cast<float>(i * COLS + j);
        }
    }
    
    float *d_column;
    cudaMalloc(&d_column, ROWS * sizeof(float));
    
    // Warmup run
    copy_first_column(h_matrix, d_column, ROWS, COLS);
    cudaDeviceSynchronize();
    
    // run a bunch of times and get average copy time.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    const int NUM_RUNS = 100;
    cudaEventRecord(start);
    for (int i = 0; i < NUM_RUNS; i++) {
        copy_first_column(h_matrix, d_column, ROWS, COLS);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "copy_first_column avg time: " << (ms * 1000) / NUM_RUNS << " us" << std::endl;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Verify
    float *h_result = new float[ROWS];
    cudaMemcpy(h_result, d_column, ROWS * sizeof(float), cudaMemcpyDeviceToHost);
    
    bool correct = true;
    for (int i = 0; i < ROWS; i++) {
        float expected = static_cast<float>(i * COLS);
        if (h_result[i] != expected) {
            std::cout << "Mismatch at " << i << ": expected " << expected << ", got " << h_result[i] << std::endl;
            correct = false;
            break;
        }
    }
    std::cout << "Verification: " << (correct ? "PASSED" : "FAILED") << std::endl;
    
    delete[] h_result;
    cudaFree(d_column);
    free(h_matrix);
    
    return 0;
}
