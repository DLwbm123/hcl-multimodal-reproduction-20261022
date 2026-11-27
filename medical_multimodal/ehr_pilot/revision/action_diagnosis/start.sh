#!/bin/bash
set -eu
umask 077
. ./env
ulimit -f 32768
dev=("$@")
for i in 0 1 2 3; do
    PART="$i" CUDA_VISIBLE_DEVICES="${dev[i]}" python -u a.py run > "a$i.log" 2>&1 &
    workers[i]=$!
done
for w in "${workers[@]}"; do wait "$w"; done
python a.py check
