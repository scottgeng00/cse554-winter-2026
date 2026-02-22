#!/bin/bash
set -ex

export CUDA_VISIBLE_DEVICES=5

PROFILER=/local1/groups/554g6/cse554-winter-2026/assignment3/cutlass/build/tools/profiler/cutlass_profiler
OUTDIR=/local1/groups/554g6/cse554-winter-2026/assignment3/Section1/cutlass_results
mkdir -p "$OUTDIR"

M_VALUES="128,256,384,512,640,768,896,1024,1152,1280,1408,1536,1664,1792,1920,2048"
SPLIT_K="1,2,4,8"

declare -a N_ARR=(512 4096 14336 4096 1024)
declare -a K_ARR=(512 4096 4096  1024 4096)

for idx in "${!N_ARR[@]}"; do
    N=${N_ARR[$idx]}
    K=${K_ARR[$idx]}
    OUTFILE="${OUTDIR}/gemm_N${N}_K${K}"

    echo "=== Profiling N=${N} K=${K} ==="
    $PROFILER \
        --operation=gemm \
        --m=$M_VALUES \
        --n=$N \
        --k=$K \
        --A=f16:column \
        --B=f16:column \
        --C=f16:column \
        --accumulator-type=f16 \
        --profiling-iterations=100 \
        --split_k_slices=$SPLIT_K \
        --split_k_mode=serial \
        --warmup-iterations=100 \
        --verification-enabled=false \
        --output="$OUTFILE"
done

echo "=== All CUTLASS profiling done ==="
