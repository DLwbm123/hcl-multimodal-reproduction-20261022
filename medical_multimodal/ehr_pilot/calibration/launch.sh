#!/bin/bash
set -eu
umask 077
cd "$(dirname "$0")"
ulimit -f 524288
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
: "${PYTHON:?}" "${PRIOR:?}" "${SOURCE:?}" "${MODEL:?}"
ARM=controls "$PYTHON" -u c.py > controls.log 2>&1
pids=()
arms=(plain old guide assisted)
for i in 0 1 2 3; do
    arm=${arms[i]}
    gpu=${GPUS:-3,5,6,7}
    IFS=, read -ra devices <<< "$gpu"
    CUDA_VISIBLE_DEVICES=${devices[i]} ARM="$arm" "$PYTHON" -u c.py > "$arm.log" 2>&1 &
    pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
test "$rc" -eq 0
ARM=summary "$PYTHON" c.py > summary.log 2>&1
test -s results.json
kill -TERM "$(cat db.pid)" || true
