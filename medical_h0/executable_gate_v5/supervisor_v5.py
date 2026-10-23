"""Independent five-second process, token and storage guard."""
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

root = Path(os.environ['RUN_ROOT'])
config = json.loads((root/'runtime_config.private.json').read_text())
sys.path[:0] = [str(root), config['identity_helpers'], config['base_code']]
from guard import identity, matches
from storage_guard import measure_usage
from budget_v5 import atomic
if (root/'SUPERVISOR_START.json').exists():
    raise RuntimeError('Refuse supervisor restart')
budget = json.loads((root/'budget.json').read_text())
atomic(root/'SUPERVISOR_START.json', identity(os.getpid()))
process = record = None
try:
    if time.time() >= budget['early_stop_epoch']:
        raise RuntimeError('Window expired before launch')
    free = int(subprocess.check_output(['nvidia-smi', '--id='+str(config['gpu']), '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
    if free < config['required_free_MiB']:
        raise RuntimeError('Insufficient GPU memory')
    used = measure_usage(root, reserve=budget['storage_reserve_bytes']) + budget['external_file_reserve_bytes']
    atomic(root/'guard_status.json', {'time': time.time(), 'bytes_with_reserve': used, 'worker': None})
    process = subprocess.Popen(['./bin/python', config['neutral_worker']], cwd=config['environment'],
        env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(config['gpu']), 'PYTHONDONTWRITEBYTECODE': '1'},
        stdout=(root/'runner.log').open('a'), stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(50):
        if (root/'WORKER_START.json').exists():
            record = json.loads((root/'WORKER_START.json').read_text()); break
        if process.poll() is not None:
            raise RuntimeError('Worker exited before registration')
        time.sleep(.1)
    if not record or record['pid'] != process.pid or not matches(record):
        raise RuntimeError('Worker identity mismatch')
    while process.poll() is None:
        used = measure_usage(root, reserve=budget['storage_reserve_bytes']) + budget['external_file_reserve_bytes']
        cost = json.loads((root/'cost_state.json').read_text()) if (root/'cost_state.json').exists() else {}
        if time.time() >= budget['early_stop_epoch'] or used >= budget['max_new_bytes']:
            raise RuntimeError('Wall-clock/storage stop')
        if cost.get('requests', 0) > budget['max_model_requests'] or cost.get('input_tokens', 0)+cost.get('reserved_input_tokens', 0) > budget['max_input_tokens'] or cost.get('output_tokens', 0)+cost.get('reserved_output_tokens', 0) > budget['max_output_tokens']:
            raise RuntimeError('Request/token budget exceeded')
        if not matches(record):
            raise RuntimeError('Worker identity changed')
        atomic(root/'guard_status.json', {'time': time.time(), 'bytes_with_reserve': used, 'worker': record, 'cost': cost})
        time.sleep(5)
    atomic(root/'SUPERVISOR_EXIT.json', {'time': time.time(), 'code': process.returncode})
except BaseException:
    atomic(root/'HARD_STOP.json', {'time': time.time(), 'error': traceback.format_exc()})
    if record and matches(record) and record['pgid'] == record['pid']:
        os.killpg(record['pid'], signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if matches(record):
                os.killpg(record['pid'], signal.SIGKILL)
    elif process and process.poll() is None:
        # This is the child just created by this supervisor, before identity
        # registration; the Popen handle still establishes ownership.
        process.terminate()
    atomic(root/'SUPERVISOR_EXIT.json', {'time': time.time(), 'code': process.poll() if process else None, 'guard_stopped': True})
    raise
