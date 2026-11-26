#!/bin/bash
set -eu
cd "$(dirname "$0")"
umask 077
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2} TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export SOURCE="${SOURCE:?set prepared benchmark folder}"
export MODEL="${MODEL:?set local model folder}"
ulimit -f 524288
remaining=$(( $(cat origin) + 22 * 3600 - $(date +%s) - 120 ))
test "$remaining" -gt 0
timeout -k 10s "$remaining" "${PYTHON:-python3}" -u pilot.py &
pid=$!
while kill -0 "$pid" 2>/dev/null; do
    if [ "$(du -sk . | cut -f1)" -gt 15000000 ]; then
        kill -TERM "$pid"
        break
    fi
    sleep 15
done
wait "$pid"
test -s aggregate.json
if [ -z "${PEER_RUN:-}" ] || [ -s "$PEER_RUN/aggregate.json" ]; then
    kill -TERM "$(cat pid)" || true
fi
