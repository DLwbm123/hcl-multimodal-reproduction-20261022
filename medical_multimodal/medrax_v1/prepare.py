"""Bounded resource preparation; no model loading or inference."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.request
import venv
import zipfile

ROOT = Path(os.environ['RUN_ROOT']).resolve()
LOCK = threading.Lock()


def save(name, data):
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def check_guard():
    guard = json.loads((ROOT / 'guard.json').read_text())
    if time.time() - guard['time'] > 90 or guard['stop']:
        raise RuntimeError('Storage/time guard unavailable or stopped')


def download(item):
    dest = ROOT / item['path']
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = item['bytes']
    if dest.exists():
        if dest.stat().st_size != size:
            raise RuntimeError('Unexpected existing asset size: ' + item['path'])
        return
    part = dest.with_name(dest.name + '.part')
    for attempt in range(3):
        check_guard()
        try:
            offset = part.stat().st_size if part.exists() else 0
            headers = {'Range': f'bytes={offset}-'} if offset else {}
            with urllib.request.urlopen(urllib.request.Request(item['url'], headers=headers), timeout=60) as response:
                append = bool(offset and response.status == 206)
                if append and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise RuntimeError('Unexpected resume range')
                written = offset if append else 0
                with part.open('ab' if append else 'wb') as target:
                    while True:
                        check_guard()
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        written += len(chunk)
                        if written > size:
                            raise RuntimeError('Response exceeds declared asset size')
                        target.write(chunk)
            if part.stat().st_size != size:
                raise RuntimeError('Incomplete HTTP transfer')
            part.replace(dest)
            with LOCK:
                with (ROOT / 'downloads.jsonl').open('a') as stream:
                    stream.write(json.dumps({'path': item['path'], 'bytes': size, 'time': time.time(),
                                             'verification': 'HTTP completion and declared size; no hash'}) + '\n')
            print('COMPLETE', item['path'], size, flush=True)
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def environment():
    env = ROOT / 'env'
    if not (env / 'pyvenv.cfg').exists():
        venv.EnvBuilder(with_pip=False, system_site_packages=True, symlinks=True).create(env)
        site = env / f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'
        shared = json.loads((ROOT / 'runtime.private.json').read_text())['read_only_site_packages']
        (site / 'shared_runtime.pth').write_text('\n'.join(shared) + '\n')
    # Keep commands neutral; paths and proxy settings arrive via the environment.
    result = subprocess.run(['./bin/python', '-m', 'pip', 'install', '--no-cache-dir', '--only-binary=:all:',
                             '--upgrade-strategy', 'only-if-needed',
                             'numpy<2', 'torchxrayvision>=1.4,<1.5', 'timm>=1.0,<1.1',
                             'einops>=0.8,<0.9', 'qwen-vl-utils>=0.0.10,<0.1'], cwd=env)
    if result.returncode:
        raise RuntimeError('Environment installation failed')
    result = subprocess.run(['./bin/python', '-m', 'pip', 'freeze'], cwd=env, capture_output=True, text=True, check=True)
    (ROOT / 'environment.freeze.txt').write_text(result.stdout)
    check = "import torch,torchvision,torchxrayvision,timm,transformers,langgraph,langchain_core,qwen_vl_utils; print('IMPORTS_OK',torch.__version__,transformers.__version__)"
    subprocess.run(['./bin/python', '-c', check], cwd=env, check=True)


def main():
    plan = json.loads((ROOT / 'code/asset_plan.json').read_text())
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(download, item): item for item in plan['files']}
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                errors.append({'path': futures[future]['path'], 'error': str(exc)[:300]})
    if errors:
        save('PREPARATION_FAILED.json', {'errors': errors, 'time': time.time(), 'model_calls': 0})
        return 1
    dataset = ROOT / 'data/chestagentbench'
    extracted = dataset / 'images'
    # Inspect names to prevent writing outside the data root. Do not use testzip/hash.
    with zipfile.ZipFile(dataset / 'figures.zip') as archive:
        for info in archive.infolist():
            check_guard()
            path = (extracted / info.filename).resolve()
            if not path.is_relative_to(extracted.resolve()):
                raise RuntimeError('Unsafe ZIP member')
            if path.exists() and (path.is_dir() or path.stat().st_size == info.file_size):
                continue
            archive.extract(info, extracted)
    environment()
    save('PREPARED.json', {'time': time.time(), 'model_calls': 0, 'status': 'ASSETS_AND_IMPORTS_READY',
                           'download_bytes': plan['download_bytes'], 'model_loaded': False,
                           'runtime_qualified': False})
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        save('PREPARATION_FAILED.json', {'error': repr(exc), 'time': time.time(), 'model_calls': 0})
        raise
