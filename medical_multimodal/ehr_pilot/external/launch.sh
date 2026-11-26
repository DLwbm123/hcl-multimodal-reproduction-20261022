#!/bin/bash
set -eu
umask 077
cd "$(dirname "$0")"
. ./env
ulimit -f 262144
export SHARDS=36
test -s controls.json
pids=()
for gpu in "$@"; do
    CUDA_VISIBLE_DEVICES="$gpu" python -u w.py > "run_$gpu.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
test "$status" = 0
python r.py
test -s result.json
kill -TERM "$(cat db.pid)" || true
