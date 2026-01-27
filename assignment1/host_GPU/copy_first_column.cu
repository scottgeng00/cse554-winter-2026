#include "copy_first_column.h"
#include <cuda_runtime.h>

void copy_first_column(float *h_A, float *d_A, int rows, int cols) {
    // Register (pin) the host matrix once for fast DMA
    static float *registered_ptr = nullptr;
    if (registered_ptr != h_A) {
        if (registered_ptr) cudaHostUnregister(registered_ptr);
        cudaHostRegister(h_A, (size_t)rows * cols * sizeof(float), cudaHostRegisterDefault);
        registered_ptr = h_A;
    }
    
    cudaMemcpy2D(
        d_A,
        sizeof(float),
        h_A,
        cols * sizeof(float),
        sizeof(float),
        rows,
        cudaMemcpyHostToDevice
    );
}
