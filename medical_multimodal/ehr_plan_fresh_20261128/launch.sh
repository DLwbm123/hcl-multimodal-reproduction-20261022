#!/usr/bin/env bash
set -euo pipefail
umask 077
. ./env
export MODEL_ROOT="$MODEL" RUN_STYLE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/cache/deps" HF_HOME="$PWD/cache/hf" TMPDIR="$PWD/cache/tmp"
export CUDA_CACHE_PATH="$PWD/cache/cuda" TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

wait_for_gpu() {
    local device=$1 minimum=$2 free
    free=$(nvidia-smi -i "$device" --query-gpu=memory.free --format=csv,noheader,nounits)
    if (( free < minimum )); then
        printf 'waiting for device %s to have %s MiB free\n' "$device" "$minimum"
        while (( free < minimum )); do
            sleep 45
            free=$(nvidia-smi -i "$device" --query-gpu=memory.free --format=csv,noheader,nounits)
        done
    fi
}

fit() {
    local device=$1 arm=$2 seed=$3 minimum=24000
    [[ "$arm" != b ]] || minimum=33000
    wait_for_gpu "$device" "$minimum"
    PHASE=fit ARM="$arm" TRIAL_SEED="$seed" CUDA_VISIBLE_DEVICES="$device" python -u t.py
}

owner=$$
pgid=$(ps -o pgid= -p "$owner" | tr -d ' ')
(
    while kill -0 "$owner" 2>/dev/null; do
        if (( $(du -sB1 . | cut -f1) >= 2500000000 )); then
            printf '%s\n' 'storage guard reached'
            kill -TERM -- "-$pgid"
            break
        fi
        sleep 30
    done
) &
watch=$!
trap 'kill "$watch" 2>/dev/null || true' EXIT
(
    fit 6 a 26112862 && fit 6 a 26112863 && fit 6 a 26112864 && fit 6 b 26112864
) > g0.log 2>&1 &
a=$!
(
    fit 7 b 26112862 && fit 7 b 26112863
) > g1.log 2>&1 &
b=$!
x=0 y=0
wait "$a" || x=$?
wait "$b" || y=$?
printf '%s\n' "$x" "$y" > fit_exits
(( x==0 && y==0 ))
wait_for_gpu 6 24000
z=0
PHASE=score CUDA_VISIBLE_DEVICES=6 python -u t.py > g2.log 2>&1 || z=$?
printf '%s\n' "$z" > exits
exit "$z"
