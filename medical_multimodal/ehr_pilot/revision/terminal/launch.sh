#!/bin/bash
set -eu
umask 077
. ./env
ulimit -f 32768
python t.py prepare
devices=("$@")
for stage in gate test; do
    workers=()
    for i in 0 1 2 3; do
        PART="$i" CUDA_VISIBLE_DEVICES="${devices[i]}" python -u t.py "$stage" > "$stage$i.log" 2>&1 &
        workers+=("$!")
    done
    status=0
    for w in "${workers[@]}"; do wait "$w" || status=1; done
    test "$status" = 0
    if test "$stage" = gate; then python t.py select; fi
done
python t.py report
test -s result.json
