call venv/scripts/activate

set HUGGINGFACE_HUB_CACHE=.\cache
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set PYTORCH_NO_CUDA_MEMORY_CACHING=1
set CUDA_LAUNCH_BLOCKING=1

python xtts_demo.py
cmd