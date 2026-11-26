#!/bin/bash
set -eu
umask 077
cd "$(dirname "$0")"
. ./env
ulimit -f 131072
python -u w.py prepare > pre.log 2>&1
for stage in train valid; do
    CUDA_VISIBLE_DEVICES="$1" WORKFLOW=5 python -u w.py "$stage" > "${stage}_a.log" 2>&1 &
    a=$!
    CUDA_VISIBLE_DEVICES="$2" WORKFLOW=9 python -u w.py "$stage" > "${stage}_b.log" 2>&1 &
    b=$!
    failure=0
    wait "$a" || failure=1
    wait "$b" || failure=1
    test "$failure" = 0
done
python -u w.py report > post.log 2>&1
test -s result.json
kill -TERM "$(cat db.pid)" || true
