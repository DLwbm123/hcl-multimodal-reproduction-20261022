"""Independent bounded process-group supervisor; persistent across SSH exit."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

root = Path(os.environ['R981_ROOT'])
budget = json.loads((root/'budget.json').read_text())
entry = Path(os.environ['NEUTRAL_WORKER'])
log = (root/'worker.log').open('a')
p = subprocess.Popen([os.environ['NEUTRAL_PYTHON'], '-u', str(entry)], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
(root/'CHILD.json').write_text(json.dumps({'pid': p.pid, 'started_epoch': budget['started_epoch']}))
reason = None
while p.poll() is None:
    used = int(subprocess.check_output(['du', '-sb', str(root)], text=True).split()[0])
    now = time.time()
    (root/'RUN_GUARD.json').write_text(json.dumps({'time': now, 'apparent_bytes': used, 'deadline': budget['hard_deadline_epoch'], 'pid': p.pid}))
    if used >= budget['stop_new_bytes'] or now >= budget['hard_deadline_epoch']:
        reason = 'storage' if used >= budget['stop_new_bytes'] else 'wall_clock'
        (root/'STOPPED.json').write_text(json.dumps({'reason': reason, 'time': now, 'apparent_bytes': used}))
        os.killpg(p.pid, signal.SIGTERM)
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
        break
    time.sleep(10)
(root/'EXIT.json').write_text(json.dumps({'time': time.time(), 'returncode': p.returncode, 'stop_reason': reason}))
