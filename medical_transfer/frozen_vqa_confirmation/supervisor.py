import os,sys,json,time,subprocess,signal,traceback
from pathlib import Path
C=Path(os.environ['CAMPAIGN_ROOT']);R=C/'runs/frozen_vqa_confirmation_v1';B=json.loads((R/'budget.json').read_text());sys.path.insert(0,str(C/'code'))
from guard import identity,matches
record={**identity(os.getpid()),'mode':'pid'};f=C/'registry'/f'{os.getpid()}.tmp';f.write_text(json.dumps(record));f.replace(f.with_suffix('.json'))
procs={};ids={}
def ready():
 s=json.loads((C/'campaign_status.json').read_text())
 if (C/'CAMPAIGN_STOP.json').exists() or time.time()-s['time']>120 or time.time()>=B['early_stop_epoch']:raise RuntimeError('campaign unavailable/deadline')
 return s
try:
 ready()
 for _ in range(35):
  if os.getpid() in ready()['live_registered']:break
  time.sleep(1)
 else:raise RuntimeError('supervisor not acknowledged')
 for role,gpu in [('baselines',2),('hcl_scoped',4)]:
  ready();out=R/role
  if (out/'START.json').exists() or (out/'EXIT.json').exists():raise RuntimeError('refuse restart')
  out.mkdir(exist_ok=True)
  free=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
  if free<35000:raise RuntimeError('insufficient GPU headroom')
  proc=subprocess.Popen([os.environ['MODEL_PYTHON'],os.environ['NEUTRAL_WORKER']],cwd=R,env={**os.environ,'RUN_ROLE':role,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTHONDONTWRITEBYTECODE':'1'},stdout=(out/'runner.log').open('a'),stderr=subprocess.STDOUT,start_new_session=True)
  procs[role]=proc;ids[role]=identity(proc.pid)
  (R/'processes.json').write_text(json.dumps({'supervisor':record,'workers':ids,'budget_started':B['started_epoch']}))
 while any(p.poll() is None for p in procs.values()):
  ready();size=max(int(subprocess.check_output(args,cwd=R,text=True,timeout=30).split()[0]) for args in [['du','-sb','.'],['du','-s','-B1','.']]);counts={}
  for role,p in procs.items():
   f=R/role/'model_attempts.json';counts[role]=json.loads(f.read_text())['attempts'] if f.exists() else 0
   if counts[role]>(B['max_calls_baselines'] if role=='baselines' else B['max_calls_per_hcl']):raise RuntimeError('call limit')
  if size>=B['early_stop_bytes']:raise RuntimeError('storage limit')
  f=R/'resource_status.tmp';f.write_text(json.dumps({'time':time.time(),'bytes':size,'calls':counts,'returncodes':{r:p.poll() for r,p in procs.items()}}));f.replace(R/'resource_status.json');time.sleep(10)
 (R/'SUPERVISOR_EXIT.json').write_text(json.dumps({'ended':time.time(),'returncodes':{r:p.returncode for r,p in procs.items()}}))
except BaseException:
 (R/'HARD_STOP.json').write_text(json.dumps({'time':time.time(),'error':traceback.format_exc()}))
 for r,p in procs.items():
  rec=ids[r]
  if p.poll() is None and matches(rec) and rec['pgid']==rec['pid']:os.killpg(p.pid,signal.SIGTERM)
 time.sleep(10)
 for r,p in procs.items():
  rec=ids[r]
  if p.poll() is None and matches(rec) and rec['pgid']==rec['pid']:os.killpg(p.pid,signal.SIGKILL)
 raise
