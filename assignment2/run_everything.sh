export CUDA_VISIBLE_DEVICES=5

mkdir -p assignment2_figs

uv run python single_batch.py
uv run python uniform_prefill.py
uv run python different_prefill.py
