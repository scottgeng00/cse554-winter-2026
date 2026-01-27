#pragma once
void rms_norm_vector(float *input, float *weight, float *output, int cols, float epsilon);

// Kernel-only version for benchmarking (no memcpy)
void rms_norm_vector_kernel_only(float *d_input, float *d_weight, float *d_output, float *d_sqsum, int cols, float epsilon);