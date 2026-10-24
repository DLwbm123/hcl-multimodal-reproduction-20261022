"""Independent clock, storage, and child-process guard for the frozen run."""
import os,json,time,signal,subprocess,traceback
from pathlib import Path
R=Path(os.environ['RUN_ROOT']);P=json.loads((R/'protocol.json').read_text());B=json.loads((R/'budget.json').read_text())
procs={};ids={}
def write(name,obj):
 p=R/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2));t.replace(p)
def identity(pid):
 p=Path('/proc')/str(pid)
 return {'pid':pid,'start_ticks':int((p/'stat').read_text().rsplit(')',1)[1].split()[19]),'pgid':os.getpgid(pid),'argv':(p/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip()}
def matches(r):
 try:return identity(r['pid'])==r
 except (FileNotFoundError,ProcessLookupError):return False
def storage():
 # This root contains every new persistent output/cache. Shared models/env/data are read-only.
 return max(int(subprocess.check_output(args,cwd=R,text=True,timeout=30).split()[0]) for args in [['du','-sb','.'],['du','-s','-B1','.']])+1048576

def stop_owned():
 for role,p in procs.items():
  if p.poll() is None and matches(ids[role]) and ids[role]['pgid']==p.pid:os.killpg(p.pid,signal.SIGTERM)
 until=time.monotonic()+10
 while time.monotonic()<until and any(p.poll() is None for p in procs.values()):time.sleep(.2)
 for role,p in procs.items():
  if p.poll() is None and matches(ids[role]) and ids[role]['pgid']==p.pid:os.killpg(p.pid,signal.SIGKILL)
 for p in procs.values():
  if p.poll() is None:p.wait(timeout=10)

def alarm(sig,frame):raise RuntimeError('independent hard deadline or termination signal')
signal.signal(signal.SIGALRM,alarm);signal.signal(signal.SIGTERM,alarm)
signal.setitimer(signal.ITIMER_REAL,max(.1,B['hard_deadline_epoch']-time.time()))
try:
 assert not (R/'LAUNCHED.json').exists(),'refuse duplicate launch'
 assert json.loads((R/'CPU_CHECK.json').read_text())['passed']
 assert time.time()<B['soft_deadline_epoch']
 write('resource_status.json',{'time':time.time(),'bytes':storage(),'calls':{},'returncodes':{}})
 write('LAUNCHED.json',{'supervisor':identity(os.getpid()),'original_start':B['started_epoch'],'launched_at':time.time()})
 for role,gpu in [('stability',2),('plasticity',3),('controls',4)]:
  free=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
  assert free>=35000,f'GPU {gpu} headroom insufficient'
  out=R/role;out.mkdir(exist_ok=True)
  assert not (out/'START.json').exists() and not (out/'EXIT.json').exists()
  proc=subprocess.Popen(['./bin/python','/data/bmw/workers/r952.py'],cwd=os.environ['MODEL_ENV'],env={**os.environ,'RUN_ROLE':role,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTHONDONTWRITEBYTECODE':'1'},stdout=(out/'runner.log').open('a'),stderr=subprocess.STDOUT,start_new_session=True)
  procs[role]=proc;ids[role]=identity(proc.pid);assert ids[role]['pgid']==proc.pid
  assert all(x not in ids[role]['argv'].lower() for x in ['wangbomin','hcl','vqa','stability','plasticity'])
  write('processes.json',{'supervisor':identity(os.getpid()),'workers':ids,'gpus':{'stability':2,'plasticity':3,'controls':4}})
 while True:
  sizes=storage();counts={};codes={r:p.poll() for r,p in procs.items()}
  for role in procs:
   f=R/role/'model_attempts.json';counts[role]=json.loads(f.read_text())['attempts'] if f.exists() else 0
   cap=P['budget']['max_calls_controls' if role=='controls' else 'max_calls_per_hcl']
   if counts[role]>cap:raise RuntimeError('call cap exceeded')
  write('resource_status.json',{'time':time.time(),'bytes':sizes,'calls':counts,'returncodes':codes})
  if any(x not in (None,0) for x in codes.values()):raise RuntimeError('worker failure; stop paired run')
  if all(x==0 for x in codes.values()):break
  if time.time()>=B['soft_deadline_epoch']:raise RuntimeError('soft deadline')
  if sizes>=P['budget']['storage_stop_bytes']:raise RuntimeError('storage guard')
  time.sleep(P['budget']['watch_interval_seconds'])
 write('SUPERVISOR_EXIT.json',{'ended':time.time(),'returncodes':codes,'original_start':B['started_epoch'],'elapsed_seconds':time.time()-B['started_epoch'],'bytes':sizes})
except BaseException:
 write('HARD_STOP.json',{'at':time.time(),'error':traceback.format_exc()});stop_owned()
 write('SUPERVISOR_EXIT.json',{'ended':time.time(),'returncodes':{r:p.poll() for r,p in procs.items()},'failed':True,'original_start':B['started_epoch']});raise
