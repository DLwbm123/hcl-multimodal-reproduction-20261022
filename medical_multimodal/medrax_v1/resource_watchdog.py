"""Apply a tighter user storage cap to existing workers without restarting them."""
import json
import os
from pathlib import Path
import signal
import time

from supervise import ROOT, storage_state


def identity(pid):
    path = Path('/proc') / str(pid)
    try:
        argv = (path / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
        started = (path / 'stat').read_text().rsplit(')', 1)[1].split()[19]
        return argv, started
    except FileNotFoundError:
        return None


def owned_groups():
    groups = {}
    policy = json.loads((ROOT / 'RESOURCE_POLICY.json').read_text())
    receipts = policy.get('launch_receipts', [str(ROOT / name) for name in [
        'native_LAUNCH.json', 'native_dependencies_LAUNCH.json',
        'supplemental_assets_LAUNCH.json', 'CONTINUATION_LAUNCH.json']])
    entries = policy.get('neutral_entries', ['/data/bmw/workers/r913.py',
        '/data/bmw/workers/r914.py', '/data/bmw/workers/r916.py'])
    for filename in receipts:
        path = Path(filename)
        if not path.exists():
            continue
        pid = json.loads(path.read_text())['pid']
        info = identity(pid)
        try:
            is_group = os.getpgid(pid) == pid
        except ProcessLookupError:
            continue
        if info and is_group:
            argv = info[0]
            try:
                cwd = (Path('/proc') / str(pid) / 'cwd').resolve(strict=True)
            except FileNotFoundError:
                continue
            own_pip = './bin/python -m pip install' in argv and str(cwd) in [
                str(Path(root) / 'env') for root in policy['roots']]
            if any(entry in argv for entry in entries) or own_pip:
                groups[pid] = info
    return groups


def main():
    while not (ROOT / 'ROUTE_DONE.json').exists():
        count, stop = storage_state()
        value = {'time': time.time(), 'bytes_with_reserve': count, 'stop': stop,
                 'hard_limit_bytes_exclusive': 30000000000, 'early_stop_bytes': 28000000000}
        path = ROOT / 'user_storage_guard.json'
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(value, indent=2) + '\n')
        temp.replace(path)
        if stop:
            groups = owned_groups()
            (ROOT / 'USER_STORAGE_HARD_STOP.json').write_text(json.dumps({**value, 'pids': list(groups)}, indent=2))
            for pid, info in groups.items():
                if identity(pid) == info:
                    try:
                        os.killpg(pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            time.sleep(5)
            for pid, info in groups.items():
                if identity(pid) == info:
                    try:
                        os.killpg(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            return 2
        time.sleep(10)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
