"""Independent clock and allocated-storage guard for one preparation/run phase."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(os.environ['RUN_ROOT']).resolve()


def allocated_bytes(root):
    total, seen = 0, set()
    for folder, dirs, names in os.walk(root, followlinks=False):
        for name in ['.'] + dirs + names:
            try:
                stat = (Path(folder) / name).lstat()
            except FileNotFoundError:
                continue
            key = (stat.st_dev, stat.st_ino)
            if key not in seen:
                total += stat.st_blocks * 512
                seen.add(key)
    return total


def storage_state():
    path = ROOT / 'RESOURCE_POLICY.json'
    policy = json.loads(path.read_text()) if path.exists() else {
        'roots': [str(ROOT)], 'reserve_bytes': 1073741824,
        'early_stop_bytes_including_reserve': 28000000000}
    count = sum(allocated_bytes(Path(p)) for p in policy['roots']) + policy['reserve_bytes'] + 1048576
    return count, count >= policy['early_stop_bytes_including_reserve']


def save(name, data):
    p = ROOT / name
    t = p.with_suffix('.tmp')
    t.write_text(json.dumps(data, indent=2) + '\n')
    t.replace(p)


def main():
    config = json.loads((ROOT / 'runtime.private.json').read_text())
    start_path = ROOT / 'FIRST_START.json'
    if not start_path.exists():
        save('FIRST_START.json', {'epoch': time.time(), 'deadline': time.time() + 23 * 3600})
    clock = json.loads(start_path.read_text())
    count, storage_stop = storage_state()
    stopped = time.time() >= clock['deadline'] or storage_stop
    save('guard.json', {'time': time.time(), 'bytes_with_reserve': count, 'stop': stopped})
    if stopped:
        raise RuntimeError('Budget unavailable before worker launch')
    with (ROOT / 'prepare.log').open('a') as log:
        child = subprocess.Popen(['./bin/python', config['prepare_entry']],
                                 cwd=config['base_environment'], stdout=log, stderr=subprocess.STDOUT,
                                 env=os.environ.copy(), start_new_session=True)
        save('PREPARE_LAUNCH.json', {'pid': child.pid, 'epoch': time.time(), 'supervisor_pid': os.getpid()})
        while child.poll() is None:
            count, storage_stop = storage_state()
            stopped = time.time() >= clock['deadline'] or storage_stop
            save('guard.json', {'time': time.time(), 'bytes_with_reserve': count, 'stop': stopped})
            if stopped:
                save('HARD_STOP.json', {'epoch': time.time(), 'bytes_with_reserve': count})
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                break
            time.sleep(15)
        code = child.wait()
    save('PREPARE_EXIT.json', {'epoch': time.time(), 'exit_code': code,
                              'allocated_bytes': allocated_bytes(ROOT), 'stopped_by_guard': stopped})
    return code


if __name__ == '__main__':
    raise SystemExit(main())
