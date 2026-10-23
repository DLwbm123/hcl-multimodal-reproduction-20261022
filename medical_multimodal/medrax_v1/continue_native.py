"""Wait for preparation, install remaining native dependencies, run bounded qualification."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from supervise import allocated_bytes, storage_state

ROOT = Path(os.environ['RUN_ROOT']).resolve()


def save(name, value):
    p = ROOT / name
    t = p.with_suffix('.tmp')
    t.write_text(json.dumps(value, indent=2) + '\n')
    t.replace(p)


def bounded(command, cwd, label, deadline, env):
    with (ROOT / (label + '.log')).open('a') as log:
        child = subprocess.Popen(command, cwd=cwd, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        save(label + '_LAUNCH.json', {'time': time.time(), 'pid': child.pid, 'supervisor_pid': os.getpid()})
        while child.poll() is None:
            size, storage_stop = storage_state()
            stop = time.time() >= deadline or storage_stop
            save('guard.json', {'time': time.time(), 'bytes_with_reserve': size, 'stop': stop})
            if stop:
                save(label + '_HARD_STOP.json', {'time': time.time(), 'bytes_with_reserve': size})
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                break
            time.sleep(5)
        code = child.wait()
    save(label + '_EXIT.json', {'time': time.time(), 'exit_code': code})
    return code


def main():
    clock = json.loads((ROOT / 'FIRST_START.json').read_text())
    save('CONTINUATION_WAITING.json', {'time': time.time(), 'deadline': clock['deadline']})
    while not (ROOT / 'PREPARE_EXIT.json').exists():
        if time.time() >= clock['deadline']:
            raise RuntimeError('Original budget exhausted while waiting for preparation')
        time.sleep(30)
    prep = json.loads((ROOT / 'PREPARE_EXIT.json').read_text())
    if prep['exit_code'] or not (ROOT / 'PREPARED.json').exists():
        raise RuntimeError('Preparation did not complete successfully')
    if 'CPU_TRANSPORT_OK' not in (ROOT / 'CPU_TRANSPORT_CHECK.txt').read_text():
        raise RuntimeError('Transport check missing')
    environment = ROOT / 'env'
    env = os.environ.copy()
    deps = ['./bin/python', '-m', 'pip', 'install', '--no-cache-dir', '--only-binary=:all:',
            '--upgrade-strategy', 'only-if-needed', 'numpy<2', 'opencv-python-headless<4.12',
            'albumentations>=1.4.24,<2', 'pyarrow>=16,<20', 'matplotlib>=3.9,<3.11', 'sentencepiece>=0.2,<0.3']
    if bounded(deps, environment, 'native_dependencies', min(clock['deadline'], time.time() + 900), env):
        raise RuntimeError('Native dependency installation failed')
    if (ROOT / 'code/supplemental_assets.json').exists():
        cutoff = min(clock['deadline'], time.time() + 900)
        if (ROOT / 'NATIVE_CLOCK.json').exists():
            cutoff = min(cutoff, json.loads((ROOT / 'NATIVE_CLOCK.json').read_text())['deadline'])
        if bounded(['./bin/python', '/data/bmw/workers/r916.py'], environment, 'supplemental_assets', cutoff, env):
            raise RuntimeError('Nested visual dependency preparation failed')
    config = json.loads((ROOT / 'runtime.private.json').read_text())
    check = subprocess.run(['nvidia-smi', '--id=' + config['gpu'], '--query-gpu=memory.free',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    free_mb = int(check.stdout.strip())
    save('NATIVE_GPU_PREFLIGHT.json', {'time': time.time(), 'gpu': config['gpu'], 'free_mib': free_mb,
                                     'required_mib': 40960})
    if free_mb < 40960:
        raise RuntimeError('Insufficient free GPU memory; existing processes left untouched')
    native_clock = ROOT / 'NATIVE_CLOCK.json'
    if native_clock.exists():
        deadline = min(clock['deadline'], json.loads(native_clock.read_text())['deadline'])
    else:
        deadline = min(clock['deadline'], time.time() + 3600)
        save('NATIVE_CLOCK.json', {'time': time.time(), 'deadline': deadline,
                                 'original_epoch': clock['epoch'], 'original_deadline': clock['deadline']})
    if time.time() >= deadline:
        raise RuntimeError('Original native phase deadline exhausted')
    env.update(CUDA_VISIBLE_DEVICES=config['gpu'], HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               MPLBACKEND='Agg', MPLCONFIGDIR=str(ROOT / 'cache/matplotlib'), NO_ALBUMENTATIONS_UPDATE='1')
    code = bounded(['./bin/python', '/data/bmw/workers/r914.py'], environment, 'native', deadline, env)
    save('CONTINUATION_FINISHED.json', {'time': time.time(), 'native_exit_code': code,
                                       'allocated_bytes': allocated_bytes(ROOT)})
    return code


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        save('CONTINUATION_FAILED.json', {'time': time.time(), 'error': repr(exc)})
        raise
