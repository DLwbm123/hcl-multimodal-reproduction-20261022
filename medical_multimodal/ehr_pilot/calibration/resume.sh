#!/bin/bash
set -eu
cd "$(dirname "$0")"
umask 077
ulimit -f 524288
export SINGLE_DECODE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
ARM=assisted "$PYTHON" -u c.py > resumed.log 2>&1
ARM=summary "$PYTHON" c.py > summary.log 2>&1
test -s results.json
kill -TERM "$(cat db.pid)" || true
