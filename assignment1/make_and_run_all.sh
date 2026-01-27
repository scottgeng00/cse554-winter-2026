#!/bin/bash

# this guy just runs everything

export CUDA_VISIBLE_DEVICES=5  # our assigned GPU!

echo "=== SiLU CUDA ==="
cd silu/CUDA && ./make_and_run.sh
cd ../..

echo ""
echo "=== RMS Norm Matrix ==="
cd rms_norm/matrix && ./make_and_run.sh
cd ../..

echo ""
echo "=== RMS Norm Vector ==="
cd rms_norm/vector && ./make_and_run.sh
cd ../..

echo ""
echo "=== Host GPU Copy ==="
cd host_GPU && ./make_and_run.sh
cd ..
