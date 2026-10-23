"""Launched through a neutral filename and environment-supplied paths."""
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path

root = Path(os.environ['RUN_ROOT'])
config = json.loads((root/'runtime_config.private.json').read_text())
sys.path[:0] = [str(root), config['v5_code'], config['transport_deps'], config['qwen_agent'], config['identity_helpers']]
from budget_v5 import atomic, BudgetStop
from guard import identity
if (root/'WORKER_START.json').exists():
    raise RuntimeError('Refuse worker restart')
record = identity(os.getpid())
assert record['pid'] == record['pgid']
atomic(root/'WORKER_START.json', record)
for sub in ('cache', 'tmp'):
    (root/sub).mkdir(exist_ok=True)
os.environ.update(HF_HOME=str(root/'cache'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
    TMPDIR=str(root/'tmp'), PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false',
    OMP_NUM_THREADS='4', CUDA_CACHE_PATH=str(root/'cache/cuda'), XDG_CACHE_HOME=str(root/'cache'),
    TORCH_HOME=str(root/'cache/torch'), TRITON_CACHE_DIR=str(root/'cache/triton'))
import tempfile
tempfile.tempdir = str(root/'tmp')


def stop(*args):
    raise BudgetStop('Independent process deadline or termination')


budget = json.loads((root/'budget.json').read_text())
signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGALRM, stop)
signal.setitimer(signal.ITIMER_REAL, max(.1, budget['hard_deadline_epoch']-time.time()))
try:
    from run_components import run
    code = run(root, config['package'], config['model_library'], config['model_path'])
except BaseException:
    atomic(root/'FAILURE.json', {'time': time.time(), 'traceback': traceback.format_exc()})
    code = 1
atomic(root/'EXIT.json', {'time': time.time(), 'code': code})
raise SystemExit(code)
