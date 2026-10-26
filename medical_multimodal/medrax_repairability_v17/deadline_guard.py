from pathlib import Path
import os,time,json,subprocess,signal
r=Path(os.environ['R1031_ROOT']);b=json.loads((r/'budget.json').read_text());seen={}
while time.time()<b['hard_deadline_epoch']+2:
 p=r/'private/active_cli.json'
 if p.exists():
  x=json.loads(p.read_text());pid=x['pid'];cmd=subprocess.run(['ps','-p',str(pid),'-o','lstart=,command='],capture_output=True,text=True).stdout.strip()
  if '/CodexCLI.app/Contents/MacOS/codex exec ' in cmd and '/tmp/r983-' in cmd:seen[pid]=cmd
 if (r/'FINISHED.json').exists():break
 time.sleep(2)
actions=[]
if time.time()>=b['hard_deadline_epoch']:
 for pid,cmd in seen.items():
  current=subprocess.run(['ps','-p',str(pid),'-o','lstart=,command='],capture_output=True,text=True).stdout.strip()
  if current and current==cmd:
   try:os.killpg(pid,signal.SIGTERM);actions.append({'pid':pid,'signal':'TERM'})
   except ProcessLookupError:pass
   time.sleep(.5)
   current=subprocess.run(['ps','-p',str(pid),'-o','lstart=,command='],capture_output=True,text=True).stdout.strip()
   if current==cmd:
    try:os.killpg(pid,signal.SIGKILL);actions.append({'pid':pid,'signal':'KILL'})
    except ProcessLookupError:pass
(r/'EXTERNAL_GUARD_EXIT.json').write_text(json.dumps({'time':time.time(),'deadline':b['hard_deadline_epoch'],'actions':actions,'observed_cli_identities':len(seen)},indent=2))
