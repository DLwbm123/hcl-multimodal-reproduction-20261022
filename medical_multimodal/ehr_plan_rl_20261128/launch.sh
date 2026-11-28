#!/usr/bin/env bash
set -euo pipefail
umask 077
. ./env
export MODEL_ROOT="$MODEL" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/cache/deps"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR="$PWD/cache/tmp"
export HF_HOME="$PWD/cache/hf" CUDA_CACHE_PATH="$PWD/cache/cuda" XDG_CACHE_HOME="$PWD/cache"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2

nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
    awk -F, '($1==6||$1==7) {if ($2<40000) exit 1; count++} END {if (count!=2) exit 1}'
owner=$$
pids=()
cleanup() {
    if (( ${#pids[@]} )); then kill -TERM "${pids[@]}" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 143' TERM
(
    while kill -0 "$owner" 2>/dev/null; do
        if (( $(du -sB1 . | cut -f1) >= 2000000000 )); then
            printf '%s\n' 'storage guard reached'
            kill -TERM "$owner"
            break
        fi
        sleep 10
    done
) &
pids+=("$!")
RUN_MODE=0 CUDA_VISIBLE_DEVICES=6 python -u t.py > a.log 2>&1 &
pids+=("$!")
RUN_MODE=1 CUDA_VISIBLE_DEVICES=7 python -u t.py > b.log 2>&1 &
pids+=("$!")
codes=()
for pid in "${pids[@]:1}"; do
    code=0
    wait "$pid" || code=$?
    codes+=("$code")
done
printf '%s\n' "${codes[@]}" > exits
(( codes[0] == 0 && codes[1] == 0 ))
