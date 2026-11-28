#!/usr/bin/env bash
set -euo pipefail
umask 077
export PATH="$PWD/j/bin:$PWD/bin:$PATH" JAVA_HOME="$PWD/j"
export PYTHONPATH="$PWD" TMPDIR="$PWD/tmp" HF_HOME="$PWD/cache/hf"
export CUDA_CACHE_PATH="$PWD/cache/cuda" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES="$GPU"
nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | awk '$1>=28000 {ok=1} END {exit !ok}'
owner=$$
(
    while kill -0 "$owner" 2>/dev/null; do
        if (( $(du -sB1 "$BUDGET_ROOT" | cut -f1) >= 1000000000 )); then
            printf '%s\n' 'storage guard reached'
            kill -TERM "$owner"
            break
        fi
        sleep 20
    done
) &
watch=$!
trap 'kill "$watch" 2>/dev/null || true' EXIT
code=0
./bin/python -u t.py || code=$?
printf '%s\n' "$code" > exits
exit "$code"
