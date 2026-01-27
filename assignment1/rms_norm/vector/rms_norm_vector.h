#pragma once
void rms_norm_vector(float *input, float *weight, float *output, int cols, float epsilon);
void rms_norm_vector_time(float *input, float *weight, int cols, float epsilon);