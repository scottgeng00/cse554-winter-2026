#pragma once
void rms_norm_matrix(float *input, float *weight, float *output, int rows, int cols, float epsilon);
void rms_norm_matrix_time(float *input, float *weight, int rows, int cols, float epsilon);