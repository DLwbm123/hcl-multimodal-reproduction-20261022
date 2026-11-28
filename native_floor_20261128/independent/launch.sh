#!/usr/bin/env bash
set -euo pipefail
umask 077
while [[ ! -f "$FULL_DIR/exits" ]]; do
    test -r "$FULL_DIR/pid"
    kill -0 "$(cat "$FULL_DIR/pid")"
    sleep 45
done
[[ $(cat "$FULL_DIR/exits") == 0 ]]
test -s "$FULL_DIR/full_result.json"
while (( $(nvidia-smi -i 4 --query-gpu=memory.free --format=csv,noheader,nounits) < 28000 )); do
    sleep 45
done
export GPU=4 BUDGET_ROOT=. HCL_PTB_LOCAL_JAR=s.jar
exec bash q.sh
