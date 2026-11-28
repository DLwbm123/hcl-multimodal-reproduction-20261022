#!/usr/bin/env bash
set -euo pipefail
umask 077
. ./env
export MODEL DATA_ROOT HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD/cache/deps" HF_HOME="$PWD/cache/hf" TMPDIR="$PWD/tmp"
export CUDA_CACHE_PATH="$PWD/cache/cuda" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
nvidia-smi -i 4 --query-gpu=memory.free --format=csv,noheader,nounits | awk '$1>=27000 {ok=1} END {exit !ok}'
owner=$$
(
    while kill -0 "$owner" 2>/dev/null; do
        if (( $(du -sB1 . | cut -f1) >= 1000000000 )); then
            printf '%s\n' 'storage guard reached'
            kill -TERM "$owner"
            break
        fi
        sleep 8
    done
) &
watch=$!
trap 'kill "$watch" 2>/dev/null || true' EXIT
code=0
CUDA_VISIBLE_DEVICES=4 python -u t.py || code=$?
printf '%s\n' "$code" > exits
exit "$code"
