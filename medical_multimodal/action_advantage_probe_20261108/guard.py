import os,json,time,signal,subprocess
from pathlib import Path
r=Path(os.environ['RUN_ROOT']);budget=json.loads((r/'budget.json').read_text());seen={};actions=[]
def identity(pid):return subprocess.run(['ps','-p',str(pid),'-o','lstart=,command='],capture_output=True,text=True).stdout.strip()
def allocated():
 total=sum(p.stat().st_blocks*512 for p in r.rglob('*') if p.is_file() and not p.is_symlink())
 for p in Path('/tmp').glob('r983-*'):
  if p.stat().st_mtime>=budget['started_epoch']-5:
   total+=sum(x.stat().st_blocks*512 for x in p.rglob('*') if x.is_file() and not x.is_symlink())
 return total
while not (r/'FINISHED.json').exists():
 try:
  p=r/'private/active_cli.json'
  if p.exists():
   pid=json.loads(p.read_text())['pid'];line=identity(pid)
   if '/CodexCLI.app/Contents/MacOS/codex exec ' in line and '/tmp/r983-' in line:seen[pid]=line
  n=allocated()
  if time.time()>=budget['hard_deadline_epoch'] or n>=budget['own_stop_new_bytes']:
   for pid,line in seen.items():
    if line and identity(pid)==line:
     try:os.killpg(pid,signal.SIGTERM);actions.append('cli_terminated')
     except ProcessLookupError:pass
   p=r/'private/worker_identity.json'
   if p.exists():
    x=json.loads(p.read_text())
    if identity(x['pid'])==x['identity']:
     try:os.killpg(x['pid'],signal.SIGTERM);actions.append('worker_terminated')
     except ProcessLookupError:pass
   break
 except Exception as e:
  actions.append('guard_error_'+type(e).__name__);break
 time.sleep(3)
(r/'GUARD_EXIT.json').write_text(json.dumps({'time':time.time(),'actions':actions}))
