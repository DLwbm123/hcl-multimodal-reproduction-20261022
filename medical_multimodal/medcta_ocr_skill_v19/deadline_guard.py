import os,json,time,signal,subprocess
from pathlib import Path
r=Path(os.environ['R18_ROOT']);b=json.loads((r/'budget.json').read_text());seen={};actions=[]
while time.time()<b['hard_deadline_epoch']:
 p=r/'private/active_cli.json'
 if p.exists():
  try:
   pid=json.loads(p.read_text())['pid'];line=subprocess.run(['ps','-p',str(pid),'-o','lstart=,command='],capture_output=True,text=True).stdout.strip()
   if '/CodexCLI.app/Contents/MacOS/codex exec ' in line and '/tmp/r983-' in line:seen[pid]=line
  except Exception:pass
 if (r/'FINISHED.json').exists():break
 time.sleep(3)
if not (r/'FINISHED.json').exists():
 for pid,line in seen.items():
  curr=subprocess.run(['ps','-p',str(pid),'-o','lstart=,command='],capture_output=True,text=True).stdout.strip()
  if curr==line:
   try:os.killpg(pid,signal.SIGTERM);actions.append('terminated_owned_cli')
   except ProcessLookupError:pass
 p=r/'private/worker_identity.json'
 if p.exists():
  x=json.loads(p.read_text());curr=subprocess.run(['ps','-p',str(x['pid']),'-o','lstart=,command='],capture_output=True,text=True).stdout.strip()
  if curr and curr==x['identity']:
   try:os.killpg(x['pid'],signal.SIGTERM);actions.append('terminated_owned_worker')
   except ProcessLookupError:pass
(r/'GUARD_EXIT.json').write_text(json.dumps({'time':time.time(),'actions':actions,'observed_cli_count':len(seen)}))
