#!/bin/bash
set -eu
umask 077
cd "$(dirname "$0")"
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
ulimit -f 524288
IFS=, read -ra devices <<< "$GPUS"
export SHARDS=${#devices[@]}
pids=()
for i in "${!devices[@]}"; do
    CUDA_VISIBLE_DEVICES=${devices[i]} PART=$i "$PYTHON" -u w.py > "$i.log" 2>&1 &
    pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
test -s result.json
kill -TERM "$(cat db.pid)" || true
