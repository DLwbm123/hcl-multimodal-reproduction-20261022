#!/usr/bin/env bash
set -euo pipefail
umask 077
. ./env
export PYTHONDONTWRITEBYTECODE=1
mkdir -p cache/tmp
export TMPDIR="$PWD/cache/tmp" XDG_CACHE_HOME="$PWD/cache" HF_HOME="$PWD/cache/hf"
export CUDA_CACHE_PATH="$PWD/cache/cuda" TORCHINDUCTOR_CACHE_DIR="$PWD/cache/torch"
own=$$
workers=()
guard=
cleanup() {
    if (("${#workers[@]}")); then
        kill -TERM "${workers[@]}" 2>/dev/null || true
    fi
    if [[ -n $guard ]]; then
        kill -TERM "$guard" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 143' TERM
(
    while kill -0 "$own" 2>/dev/null; do
        if (( $(du -sB1 . | cut -f1) >= 200000000 )); then
            echo "storage guard reached"
            kill -TERM "$own"
            break
        fi
        sleep 5
    done
) &
guard=$!

CUDA_VISIBLE_DEVICES="" python -u t.py control > control.log 2>&1 &
workers=("$!")
wait "${workers[0]}"
workers=()
for part in 0 1; do
    CUDA_VISIBLE_DEVICES=$((6 + part)) PART=$part python -u t.py infer > "$part.log" 2>&1 &
    workers+=("$!")
done
for pid in "${workers[@]}"; do
    wait "$pid"
done
workers=()
CUDA_VISIBLE_DEVICES="" python -u t.py report > report.log 2>&1 &
workers=("$!")
wait "${workers[0]}"
workers=()
