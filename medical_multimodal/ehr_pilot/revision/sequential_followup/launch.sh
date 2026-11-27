#!/bin/bash
set -eu
umask 077
. ./env
ulimit -f 32768
python t.py prepare
devices=("$@")
workers=()
for i in 0 1 2 3; do
    PART="$i" CUDA_VISIBLE_DEVICES="${devices[i]}" python -u t.py eval > "eval$i.log" 2>&1 &
    workers+=("$!")
done
status=0
for w in "${workers[@]}"; do wait "$w" || status=1; done
test "$status" = 0
test -s result.json
