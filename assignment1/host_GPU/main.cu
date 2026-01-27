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
    
    copy_first_column_time(h_matrix, ROWS, COLS);
    
    delete[] h_result;
    cudaFree(d_column);
    free(h_matrix);
    
    return 0;
}
