"""Independent hard clock and allocated-storage enforcement for one worker."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from storage_guard import allocated_bytes

ROOT = Path(os.environ['R1136_ROOT'])
cfg = json.loads((ROOT/'deployment.private.json').read_text())
budget = json.loads((ROOT/'budget.json').read_text())


def save(name, value):
    path = ROOT/name
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


env = os.environ.copy()
env.update(R981_ROOT=str(ROOT), CUDA_VISIBLE_DEVICES=str(cfg['gpu']),
           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
           HF_HOME=cfg['neutral_cache'], HF_MODULES_CACHE=str(Path(cfg['neutral_cache'])/'modules'),
           TMPDIR=cfg['neutral_tmp'], R1136_TMP=cfg['neutral_tmp'], R1136_CACHE=cfg['neutral_cache'],
           PYTHONDONTWRITEBYTECODE='1', NO_ALBUMENTATIONS_UPDATE='1',
           PYTHONPATH=str(ROOT/'code'))
with (ROOT/'worker.log').open('a') as log, (ROOT/'code/worker.py').open() as source:
    process = subprocess.Popen([cfg['neutral_python'], '-u', '-'], stdin=source, stdout=log,
                               stderr=subprocess.STDOUT, cwd='/data/bmw/workers', env=env, start_new_session=True)
save('WORKER.json', {'pid': process.pid, 'started': time.time(), 'gpu': cfg['gpu']})
reason = None
while process.poll() is None:
    storage_root = Path(cfg.get('storage_root', str(ROOT)))
    used = allocated_bytes([storage_root])
    save('GUARD.json', {'time': time.time(), 'allocated_bytes': used, 'deadline': budget['hard_deadline_epoch'],
                        'worker_pid': process.pid})
    if used >= budget['remote_storage_stop_bytes'] or time.time() >= budget['hard_deadline_epoch']:
        reason = 'storage' if used >= budget['remote_storage_stop_bytes'] else 'deadline'
        save('STOPPED.json', {'time': time.time(), 'reason': reason, 'allocated_bytes': used})
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        break
    time.sleep(5)
save('EXIT.json', {'time': time.time(), 'worker_exit_code': process.returncode, 'guard_stop_reason': reason})
