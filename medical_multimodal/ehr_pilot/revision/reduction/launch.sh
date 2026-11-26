#!/bin/bash
set -eu
umask 077
. ./env
ulimit -f 32768
python prep.py
for n in $(seq 1 90); do
  if curl --noproxy 127.0.0.1 -fsS --max-time 3 http://127.0.0.1:18478/fhir/metadata -o /dev/null 2>/dev/null; then break; fi
  sleep 2
done
python i.py
python t.py build
children=()
for part in 0 1 2 3; do
    device=$1
    shift
    CUDA_VISIBLE_DEVICES="$device" PART="$part" python -u t.py infer > "$part.log" 2>&1 &
    children+=("$!")
done
status=0
for child in "${children[@]}"; do wait "$child" || status=1; done
test "$status" = 0
python t.py report
test -s result.json
kill -TERM "$(cat db.pid)" || true
