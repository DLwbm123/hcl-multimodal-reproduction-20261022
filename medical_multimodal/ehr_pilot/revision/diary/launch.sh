#!/bin/bash
set -eu
umask 077
. ./env
ulimit -f 32768
python engine.py
devices=("$@")
for phase in build gate test; do
    if test "$phase" = gate; then CUDA_VISIBLE_DEVICES="${devices[0]}" python -u t.py edit; fi
    workers=()
    for i in 0 1 2 3; do
        PART="$i" CUDA_VISIBLE_DEVICES="${devices[i]}" python -u t.py "$phase" > "$phase$i.log" 2>&1 &
        workers+=("$!")
    done
    st=0
    for pid in "${workers[@]}"; do wait "$pid" || st=1; done
    test "$st" = 0
done
python t.py report
test -s result.json
