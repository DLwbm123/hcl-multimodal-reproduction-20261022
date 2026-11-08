import os,json,time,signal,subprocess
from pathlib import Path
r=Path(os.environ['RUN_ROOT']);budget=json.loads((r/'budget.json').read_text());seen={};reason=None;actions=[]
def ident(pid):return subprocess.run(['ps','-p',str(pid),'-o','lstart=,command='],capture_output=True,text=True).stdout.strip()
def allocated():
 total=sum(p.stat().st_blocks*512 for p in r.rglob('*') if p.is_file() and not p.is_symlink())
 for p in Path('/tmp').glob('r983-*'):
  try:
   if p.stat().st_mtime>=budget['started_epoch']-5:total+=sum(x.stat().st_blocks*512 for x in p.rglob('*') if x.is_file() and not x.is_symlink())
  except FileNotFoundError:pass
 return total
while not (r/'FINISHED.json').exists():
 try:
  p=r/'private/active_cli.json'
  if p.exists():
   pid=json.loads(p.read_text())['pid'];line=ident(pid)
   if '/CodexCLI.app/Contents/MacOS/codex exec ' in line and '/tmp/r983-' in line:seen[pid]=line
  if time.time()>=budget['hard_deadline_epoch']:reason='deadline';break
  if allocated()>=budget['own_stop_new_bytes']:reason='storage';break
 except Exception as e:reason='guard_error_'+type(e).__name__;break
 time.sleep(2)
if reason:
 for pid,line in seen.items():
  if line and ident(pid)==line:
   try:os.killpg(pid,signal.SIGTERM);actions.append('cli_terminated')
   except ProcessLookupError:pass
 x=json.loads((r/'private/worker_identity.json').read_text())
 if ident(x['pid'])==x['identity']:
  try:os.killpg(x['pid'],signal.SIGTERM);actions.append('worker_terminated')
  except ProcessLookupError:pass
(r/'GUARD2_EXIT.json').write_text(json.dumps({'time':time.time(),'reason':reason,'actions':actions}))
