set -ex

export CUDA_VISIBLE_DEVICES=5

nvcc -O3 -arch=sm_75 -lcublas Section1/prof_cublas_gemm.cu -o Section1/prof_cublas_gemm

./Section1/prof_cublas_gemm

uv run python Section1/plot_gemm.py
