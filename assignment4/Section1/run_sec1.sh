export CUDA_VISIBLE_DEVICES=5

echo "=== Naive vs Continuous: uniform input [1,10], output [1,128], bs=10, 100 reqs ==="
uv run python Section1/profiler_code.py \
    --methods naive continuous \
    --num-requests 100 --batch-size 10 \
    --input-dist uniform --min-input-len 1 --max-input-len 10 \
    --min-output-len 1 --max-output-len 128

echo ""
echo "=== Chunked vs Continuous: lognormal input (μ=6,σ=0.7), output [1,512], 100 reqs ==="
uv run python Section1/profiler_code.py \
    --methods continuous chunked \
    --num-requests 100 --batch-size 48 --token-budget 512 \
    --input-dist lognormal --input-mean 6.0 --input-sigma 0.7 \
    --min-output-len 1 --max-output-len 512 \
    --csv iter_chunked_vs_cont.csv

echo ""
echo "=== Plotting iteration times ==="
uv run python Section1/plot_iterations.py iter_chunked_vs_cont.csv -o iteration_times.pdf


echo ""
echo "=== vLLM benchmark: input=512, output=512, 100 reqs ==="
uv run vllm bench throughput \
    --model /local1/cse554/models/meta-llama/Llama-3.2-1B \
    --dataset-name random \
    --random-input-len 512 \
    --random-output-len 512 \
    --num-prompts 100

echo ""
echo "=== Chunked prefill: input=512, output=512, 100 reqs ==="
uv run python Section1/profiler_code.py \
    --methods chunked \
    --num-requests 100 --token-budget 8192 \
    --input-dist uniform --min-input-len 512 --max-input-len 512 \
    --min-output-len 512 --max-output-len 512

echo ""
echo "=== Kernel breakdown: our chunked engine ==="
uv run python Section1/profile_kernels.py

echo ""
echo "=== Kernel breakdown: vLLM (via nsys) ==="
nsys profile --stats=true -o vllm_profile -f true \
    uv run vllm bench throughput \
    --model /local1/cse554/models/meta-llama/Llama-3.2-1B \
    --dataset-name random \
    --random-input-len 512 --random-output-len 512 \
    --num-prompts 20