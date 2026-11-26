#!/bin/bash
set -eu
umask 077
cd "$(dirname "$0")"
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export SHARDS=5
bash s.sh > db.log 2>&1 &
db=$!
ulimit -f 524288
trap 'kill "$db" 2>/dev/null || true' EXIT
for ((i=0;i<120;i++)); do
    kill -0 "$db"
    if curl -fsS --max-time 3 "${FHIR_BASE}metadata" -o /dev/null 2>/dev/null; then break; fi
    sleep 2
done
curl -fsS --max-time 8 "${FHIR_BASE}metadata" -o /dev/null
IFS=, read -ra devices <<< "$GPUS"
test "${#devices[@]}" = "$SHARDS"
pids=()
for i in "${!devices[@]}"; do
    CUDA_VISIBLE_DEVICES=${devices[i]} PART=$i "$PYTHON" -u v.py > "$i.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
