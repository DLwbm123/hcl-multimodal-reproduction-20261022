#!/bin/bash
set -eu
umask 077
cd "$(dirname "$0")"
. ./env
ulimit -f 32768
python b.py
python i.py
python w.py control
pids=()
for i in 0 1 2; do
    g=$1;shift
    CUDA_VISIBLE_DEVICES="$g" PART="$i" python -u w.py infer > "$i.log" 2>&1 &
    pids+=("$!")
done
status=0; for p in "${pids[@]}"; do wait "$p" || status=1; done
test "$status" = 0
python w.py report
test -s result.json
kill -TERM "$(cat db.pid)" || true
