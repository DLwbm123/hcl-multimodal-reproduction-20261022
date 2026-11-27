#!/usr/bin/env bash
set -euo pipefail
umask 077
. ./env
export MODEL_ROOT="$MODEL" PYTHONDONTWRITEBYTECODE=1
mkdir -p cache/tmp
export TMPDIR="$PWD/cache/tmp" XDG_CACHE_HOME="$PWD/cache" HF_HOME="$PWD/cache/hf"
export CUDA_CACHE_PATH="$PWD/cache/cuda"
own=$$
pids=()
cleanup() {
    if (("${#pids[@]}")); then kill -TERM "${pids[@]}" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 143' TERM
(
    while kill -0 "$own" 2>/dev/null; do
        if (( $(du -sB1 . | cut -f1) >= 300000000 )); then
            printf '%s\n' "storage guard reached"
            kill -TERM "$own"
            break
        fi
        sleep 5
    done
) &
pids+=("$!")
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
    awk -F, '($1==6||$1==7) {if ($2<22000) exit 1; count++} END {if (count!=2) exit 1}'
for panel in a b; do
    if [[ $panel == a ]]; then gpu=6; mode=visual; else gpu=7; mode=text; fi
    (
        cd "$panel"
        RUN_ROOT="$PWD" DATA_ROOT="$PWD" CUDA_VISIBLE_DEVICES="$gpu" MODE="$mode" exec python -u ../t.py
    ) > "$panel.log" 2>&1 &
    pids+=("$!")
done
wait "${pids[1]}"
wait "${pids[2]}"
