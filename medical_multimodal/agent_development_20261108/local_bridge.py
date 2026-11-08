"""Three independent queues, one locked controller-call budget and hard clock."""
import concurrent.futures
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time
from storage_guard import allocated_bytes

ROOT = Path(os.environ['R1136_LOCAL'])
cfg = json.loads((ROOT/'deployment.private.json').read_text())
spec = importlib.util.spec_from_file_location('subscription', ROOT/'code/subscription.py')
subscription = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subscription)


def rpc(partition, action, payload=None):
    proc = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'pro5000',
                           'python3', partition['rpc_path']], input=json.dumps({'action': action, 'payload': payload}),
                          text=True, capture_output=True, timeout=45)
    if proc.returncode:
        raise RuntimeError('Queue RPC failed: '+proc.stderr[-500:])
    return json.loads(proc.stdout)


def consume(partition, budget):
    while time.time() < budget['hard_deadline_epoch']:
        used = allocated_bytes([ROOT, Path(budget['local_temp_root'])])
        if used >= budget['local_storage_stop_bytes']:
            (ROOT/'LOCAL_STOPPED.json').write_text(json.dumps({'time': time.time(), 'reason': 'storage'}))
            rpc(partition, 'stop', {'reason': 'shared local storage guard'})
            return
        try:
            job = rpc(partition, 'pull')
            if job['state'] == 'finished':
                return
            if job['state'] != 'job':
                time.sleep(3)
                continue
            path = ROOT/'private'/('response_'+job['key']+'.json')
            if path.exists():
                response = json.loads(path.read_text())
            else:
                response = subscription.execute(job['job'], ROOT)
                tmp = path.with_suffix('.tmp')
                tmp.write_text(json.dumps(response))
                tmp.replace(path)
            rpc(partition, 'respond', {'key': job['key'], 'response': response})
            print(json.dumps({'partition': partition['name'], 'key': job['key'], 'status': response['status'],
                              'seconds': response.get('seconds')}), flush=True)
        except Exception as exc:
            print(json.dumps({'time': time.time(), 'partition': partition['name'], 'error': repr(exc)}), flush=True)
            time.sleep(10)


def main():
    budget = json.loads((ROOT/'budget.json').read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(consume, p, budget) for p in cfg['partitions']]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    (ROOT/'BRIDGE_FINISHED.json').write_text(json.dumps({'time': time.time()}))


if __name__ == '__main__':
    main()
