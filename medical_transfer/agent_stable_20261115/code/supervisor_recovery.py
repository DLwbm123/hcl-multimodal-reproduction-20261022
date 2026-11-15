"""One independent deadline and storage guard across learning and all controls."""
import os,json,time,signal,subprocess,traceback,resource
from pathlib import Path
R=Path(os.environ['RUN_ROOT']);P=json.loads((R/'protocol.json').read_text());B=json.loads((R/'budget.json').read_text());procs={};ids={};gpu_for={}
def write(n,x):
 p=R/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def identity(pid):
 p=Path('/proc')/str(pid)
 return {'pid':pid,'start_ticks':int((p/'stat').read_text().rsplit(')',1)[1].split()[19]),'pgid':os.getpgid(pid),'argv':(p/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip()}
def matches(r):
 try:return identity(r['pid'])==r
 except (FileNotFoundError,ProcessLookupError):return False
import sys
sys.path.insert(0,str(R/'code'))
from storage_guard import measure
def storage():return measure(R)
def stop():
 for r,p in procs.items():
  if p.poll() is None and matches(ids[r]) and ids[r]['pgid']==p.pid:os.killpg(p.pid,signal.SIGTERM)
 until=time.monotonic()+10
 while time.monotonic()<until and any(p.poll() is None for p in procs.values()):time.sleep(.2)
 for r,p in procs.items():
  if p.poll() is None and matches(ids[r]) and ids[r]['pgid']==p.pid:os.killpg(p.pid,signal.SIGKILL)
 for p in procs.values():
  if p.poll() is None:p.wait(timeout=10)
def alarm(*args):raise RuntimeError('independent deadline or termination')
signal.signal(signal.SIGALRM,alarm);signal.signal(signal.SIGTERM,alarm);signal.setitimer(signal.ITIMER_REAL,max(.1,B['hard_deadline_epoch']-time.time()))
resource.setrlimit(resource.RLIMIT_FSIZE,(256*1024**2,256*1024**2))
def status(phase):
 size=storage();counts={};codes={r:p.poll() for r,p in procs.items()}
 for r in procs:
  f=R/r/'model_attempts.json';counts[r]=json.loads(f.read_text())['attempts'] if f.exists() else 0
  if counts[r]>P['budget']['max_calls_per_worker']:raise RuntimeError('call cap exceeded')
 write('resource_status.json',{'time':time.time(),'phase':phase,'bytes':size,'calls':counts,'returncodes':codes})
 if any(c not in (None,0) for c in codes.values()):raise RuntimeError('worker failure; paired study stopped')
 if time.time()>=B['soft_deadline_epoch']:raise RuntimeError('original soft deadline')
 if size>=P['budget']['storage_stop_bytes']:raise RuntimeError('storage early stop')
 return codes

def free(gpu):return int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
def launch(role,gpu):
 assert free(gpu)>=P['budget']['gpus']['minimum_free_mib'],'GPU headroom'
 out=R/role;out.mkdir(exist_ok=True);assert not (out/'START.json').exists() and not (out/'EXIT.json').exists()
 p=subprocess.Popen([os.environ['WORKER_PYTHON'],os.environ['WORKER_ENTRY']],env={**os.environ,'RUN_ROLE':role,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTHONDONTWRITEBYTECODE':'1'},cwd=R,stdout=(out/'runner.log').open('a'),stderr=subprocess.STDOUT,start_new_session=True)
 procs[role]=p;ids[role]=identity(p.pid);gpu_for[role]=gpu;assert ids[role]['pgid']==p.pid
 assert all(x not in ids[role]['argv'].lower() for x in os.environ['FORBIDDEN_PROCESS_TERMS'].split(','))
 write('processes.json',{'supervisor':identity(os.getpid()),'workers':ids,'gpus':gpu_for})
try:
 assert json.loads((R/'CPU_CHECK.json').read_text())['passed'];assert json.loads((R/'AGENT_CHECK.json').read_text())['passed']
 assert (R/'RECOVERY_PLAN.json').exists()
 assert json.loads((R/'FROZEN_SELECTION.json').read_text())['locked_before_confirmation']
 for role in ['fresh','resume']:assert json.loads((R/role/'EXIT.json').read_text())['exit_code']==0
 assert not (R/'RECOVERY_LAUNCHED.json').exists()
 write('RECOVERY_LAUNCHED.json',{'supervisor':identity(os.getpid()),'original_start':B['started_epoch'],'at':time.time()})
 status('confirmation')
 queued=['confirm_'+a for a in P['confirmation']['arms']]
 while queued or any(p.poll() is None for p in procs.values()):
  status('confirmation');busy={gpu_for[r] for r,p in procs.items() if p.poll() is None}
  for gpu in P['budget']['gpus']['confirmation_parallel_candidates']:
   if not queued or len(busy)>=P['budget']['gpus']['max_confirmation_workers']:break
   if gpu not in busy and free(gpu)>=P['budget']['gpus']['minimum_free_mib']:
    launch(queued.pop(0),gpu);busy.add(gpu)
  time.sleep(P['budget']['watch_interval_seconds'])
 status('analysis')
 for r in procs:assert json.loads((R/r/'EXIT.json').read_text())['exit_code']==0
 import sys
 sys.path.insert(0,str(R/'code'))
 from analyze import analyze
 analyze(R)
 write('SUPERVISOR_EXIT.json',{'ended':time.time(),'returncodes':{r:p.poll() for r,p in procs.items()},'elapsed_seconds':time.time()-B['started_epoch'],'bytes':storage(),'success':True})
except BaseException:
 write('HARD_STOP.json',{'at':time.time(),'error':traceback.format_exc()});stop();write('SUPERVISOR_EXIT.json',{'ended':time.time(),'success':False,'returncodes':{r:p.poll() for r,p in procs.items()},'original_start':B['started_epoch']});raise
