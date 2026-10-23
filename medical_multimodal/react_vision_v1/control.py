"""One-hour independent diagnostic within the existing route clock/storage ceiling."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.append(str(Path(__file__).resolve().parents[1] / 'medrax_v1'))
from continue_native import bounded, save
from supervise import storage_state

ROOT = Path(os.environ['RUN_ROOT']).resolve()


def main():
    plan = json.loads((ROOT / 'code/protocol.json').read_text())
    config = json.loads((ROOT / 'runtime.private.json').read_text())
    clock = ROOT / 'FIRST_START.json'
    if not clock.exists():
        now = time.time()
        save('FIRST_START.json', {'epoch': now, 'deadline': min(now + plan['phase_seconds'], plan['original_route_deadline'])})
    deadline = json.loads(clock.read_text())['deadline']
    size, stop = storage_state()
    save('guard.json', {'time': time.time(), 'bytes_with_reserve': size, 'stop': stop})
    if stop or time.time() >= deadline:
        raise RuntimeError('Experiment budget unavailable before launch')
    free = int(subprocess.check_output(['nvidia-smi', '--id=' + config['gpu'],
        '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
    save('GPU_PREFLIGHT.json', {'time': time.time(), 'gpu': config['gpu'], 'free_mib': free})
    if free < 51200:
        raise RuntimeError('Insufficient free GPU memory')
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=config['gpu'], HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       MPLBACKEND='Agg', MPLCONFIGDIR=str(ROOT / 'cache/matplotlib'), NO_ALBUMENTATIONS_UPDATE='1')
    code = bounded(['./bin/python', '/data/bmw/workers/r942.py'], config['environment'], 'paired', deadline, environment)
    save('CONTROL_EXIT.json', {'time': time.time(), 'exit_code': code})
    return code


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        save('CONTROL_FAILED.json', {'time': time.time(), 'error': repr(exc)})
        raise
