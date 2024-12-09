@echo off

call venv/scripts/activate

set HUGGINGFACE_HUB_CACHE=.\cache

python xtts_demo.py
cmd