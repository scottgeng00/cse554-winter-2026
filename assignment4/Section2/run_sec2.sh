export CUDA_VISIBLE_DEVICES=5,6

uv run python Section2/profile_allreduce.py


uv run python Section2/transformer-w3l1.py


uv run torchrun --nproc-per-node=2 Section2/transformer-tp2.py