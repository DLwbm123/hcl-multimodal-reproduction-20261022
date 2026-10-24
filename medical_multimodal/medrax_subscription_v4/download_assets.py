"""Bounded downloads from authorized, short-lived URLs; no account token on server."""
import concurrent.futures
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import urllib.request
import urllib.parse

ROOT = Path(os.environ['R981_ROOT'])
PLAN = json.loads((ROOT / 'private/download.json').read_text())

def guard():
    while True:
        used = int(subprocess.check_output(['du', '-sb', str(ROOT)]).split()[0])
        state = {'epoch': time.time(), 'bytes': used, 'deadline': PLAN['hard_deadline_epoch']}
        (ROOT / 'guard.json').write_text(json.dumps(state))
        if used >= PLAN['stop_new_bytes'] or time.time() >= PLAN['hard_deadline_epoch']:
            (ROOT / 'STOPPED.json').write_text(json.dumps(state))
            os.killpg(os.getpgrp(), signal.SIGTERM)
        time.sleep(15)

def fetch(item):
    target = ROOT / 'models' / item['directory'] / item['name']
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == item['size']:
        return item['name']
    part = target.with_name(target.name + '.partial')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({'https': 'http://127.0.0.1:17897'}))
    failures=0
    while (part.stat().st_size if part.exists() else 0) < item['size']:
        offset=part.stat().st_size if part.exists() else 0
        end=min(offset+64*1024*1024,item['size'])-1
        try:
            fresh=json.loads((ROOT/'private/download.json').read_text())
            url=next(f['url'] for f in fresh['files'] if f['directory']==item['directory'] and f['name']==item['name'])
            url=urllib.parse.urljoin('https://huggingface.co/',url)
            with opener.open(urllib.request.Request(url,headers={'Range':f'bytes={offset}-{end}'}),timeout=90) as src:
                if src.status!=206 or not src.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                    # Tiny non-LFS files may ignore Range and return their complete body.
                    if not (offset==0 and item['size']<1024*1024 and src.status==200):raise RuntimeError('range response mismatch')
                with part.open('ab' if offset else 'wb') as out:
                    remaining=end-offset+1
                    while remaining:
                        if time.time()>=PLAN['hard_deadline_epoch']:raise RuntimeError('deadline')
                        chunk=src.read(min(4*1024*1024,remaining))
                        if not chunk:raise RuntimeError('short response')
                        out.write(chunk);remaining-=len(chunk)
            failures=0
        except Exception as exc:
            failures+=1
            print(json.dumps({'file':item['name'],'consecutive_failures':failures,'offset':part.stat().st_size if part.exists() else 0,'error_type':type(exc).__name__,'status':getattr(exc,'code',None)}),flush=True)
            if failures>=6:raise RuntimeError('download failed: '+item['name']) from None
            time.sleep(3)
    if part.stat().st_size!=item['size']:raise RuntimeError('length mismatch')
    part.rename(target)
    print(json.dumps({'completed':item['directory']+'/'+item['name'],'bytes':item['size']}),flush=True)
    return item['name']

if __name__ == '__main__':
    threading.Thread(target=guard, daemon=True).start()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(fetch, PLAN['files']))
        (ROOT / 'ASSETS_READY.json').write_text(json.dumps({'time': time.time(), 'files': len(PLAN['files']), 'expected_bytes': sum(x['size'] for x in PLAN['files'])}))
    finally:
        (ROOT / 'DOWNLOAD_EXIT.json').write_text(json.dumps({'time': time.time(), 'ready': (ROOT/'ASSETS_READY.json').exists()}))
