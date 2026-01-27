#pragma once
void rms_norm_matrix(float *input, float *weight, float *output, int rows, int cols, float epsilon);

// Kernel-only version for benchmarking (no memcpy)
void rms_norm_matrix_kernel_only(float *d_input, float *d_weight, float *d_output, int rows, int cols, float epsilon);