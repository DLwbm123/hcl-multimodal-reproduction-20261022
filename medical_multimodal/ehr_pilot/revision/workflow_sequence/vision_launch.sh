#!/usr/bin/env bash
set -euo pipefail
umask 077
. ./env
export PYTHONDONTWRITEBYTECODE=1
mkdir -p cache/tmp s0 s1
export TMPDIR="$PWD/cache/tmp" XDG_CACHE_HOME="$PWD/cache" HF_HOME="$PWD/cache/hf"
export CUDA_CACHE_PATH="$PWD/cache/cuda" TORCHINDUCTOR_CACHE_DIR="$PWD/cache/torch"
export DATA_ROOT="$PWD/old_data" RUN_ROOT="$PWD" MODEL_ROOT="$MODEL"
own=$$
pids=()
cleanup() {
    if (("${#pids[@]}")); then kill -TERM "${pids[@]}" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 143' TERM
(
    while kill -0 "$own" 2>/dev/null; do
        if (( $(du -sB1 . | cut -f1) >= 500000000 )); then
            echo "storage guard reached"
            kill -TERM "$own"
            break
        fi
        sleep 5
    done
) &
pids+=("$!")
if [[ ! -s composition.json ]]; then
    CUDA_VISIBLE_DEVICES="" python -u t.py freeze > freeze.log 2>&1 &
    pids+=("$!")
    wait "${pids[-1]}"
    unset 'pids[-1]'
fi
# Existing GPU workers from the preceding phase may share these same two devices.
for tick in $(seq 1 240); do
    prior_pid=$(cat previous/pid)
    if ! kill -0 "$prior_pid" 2>/dev/null; then break; fi
    sleep 5
done
! kill -0 "$(cat previous/pid)" 2>/dev/null
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
    awk -F, '($1==6||$1==7) {if ($2<22000) exit 1; count++} END {if (count!=2) exit 1}'
for part in 0 1; do
    ln -sfn ../design.json "s$part/design.json"
    ln -sfn ../cases.json "s$part/cases.json"
    (
        cd "s$part"
        RUN_ROOT="$PWD" CUDA_VISIBLE_DEVICES=$((6 + part)) PART=$part exec python -u ../t.py infer
    ) > "$part.log" 2>&1 &
    pids+=("$!")
done
wait "${pids[1]}"
wait "${pids[2]}"
CUDA_VISIBLE_DEVICES="" python -u t.py report > report.log 2>&1 &
pids+=("$!")
wait "${pids[3]}"
