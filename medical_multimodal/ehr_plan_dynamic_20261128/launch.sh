#!/usr/bin/env bash
set -euo pipefail
umask 077
. ./env
export MODEL_ROOT="$MODEL" RUN_STYLE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/cache/deps" HF_HOME="$PWD/cache/hf" TMPDIR="$PWD/cache/tmp"
export CUDA_CACHE_PATH="$PWD/cache/cuda" TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p a b
nvidia-smi -i 6,7 --query-gpu=index,memory.free --format=csv,noheader,nounits |
    awk -F, '$2 < 30000 {bad=1} END {exit bad || NR != 2}'
owner=$$
pids=()
cleanup() {
    if (( ${#pids[@]} )); then kill -TERM "${pids[@]}" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 143' TERM
(
    while kill -0 "$owner" 2>/dev/null; do
        if (( $(du -scB1 a b cache | tail -n 1 | cut -f1) >= 1000000000 )); then
            printf '%s\n' 'storage guard reached'
            kill -TERM "$owner"
            break
        fi
        sleep 8
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
