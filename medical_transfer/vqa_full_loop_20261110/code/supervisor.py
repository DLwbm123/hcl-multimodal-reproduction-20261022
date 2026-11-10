"""Single original clock and independent process/storage guard across both phases."""
import os,json,time,signal,subprocess,traceback,resource
from pathlib import Path
R=Path(os.environ['RUN_ROOT']);P=json.loads((R/'protocol.json').read_text());B=json.loads((R/'budget.json').read_text());procs={};ids={}
def write(n,x):
 p=R/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def identity(pid):
 p=Path('/proc')/str(pid)
 return {'pid':pid,'start_ticks':int((p/'stat').read_text().rsplit(')',1)[1].split()[19]),'pgid':os.getpgid(pid),'argv':(p/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip()}
def matches(r):
 try:return identity(r['pid'])==r
 except (FileNotFoundError,ProcessLookupError):return False
def storage():
 return max(int(subprocess.check_output(a,cwd=R,text=True,timeout=30).split()[0]) for a in [['du','-sb','.'],['du','-s','-B1','.']])+1048576
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
# No generated single file can exceed 256 MB. All caches and writes stay under R.
resource.setrlimit(resource.RLIMIT_FSIZE,(256*1024**2,256*1024**2))
def status(phase):
 sizes=storage();counts={};codes={r:p.poll() for r,p in procs.items()}
 for r in procs:
  f=R/r/'model_attempts.json';counts[r]=json.loads(f.read_text())['attempts'] if f.exists() else 0
  if counts[r]>P['budget']['max_calls_per_worker']:raise RuntimeError('call cap exceeded')
 write('resource_status.json',{'time':time.time(),'phase':phase,'bytes':sizes,'calls':counts,'returncodes':codes})
 if any(c not in (None,0) for c in codes.values()):raise RuntimeError('worker failure; paired study stopped')
 if time.time()>=B['soft_deadline_epoch']:raise RuntimeError('original soft deadline')
 if sizes>=P['budget']['storage_stop_bytes']:raise RuntimeError('storage limit')
 return codes

def run_phase(phase,assignments):
 status(phase)
 for role,gpu in assignments:
  free=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>=35000,'GPU headroom'
  out=R/role;out.mkdir(exist_ok=True);assert not (out/'START.json').exists() and not (out/'EXIT.json').exists()
  p=subprocess.Popen([os.environ['WORKER_PYTHON'],os.environ['WORKER_ENTRY']],env={**os.environ,'RUN_ROLE':role,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTHONDONTWRITEBYTECODE':'1'},cwd=R,stdout=(out/'runner.log').open('a'),stderr=subprocess.STDOUT,start_new_session=True)
  procs[role]=p;ids[role]=identity(p.pid);assert ids[role]['pgid']==p.pid
  assert all(x not in ids[role]['argv'].lower() for x in os.environ.get('FORBIDDEN_PROCESS_TERMS','hcl,vqa,fresh,resume').split(','))
  write('processes.json',{'supervisor':identity(os.getpid()),'workers':ids})
 while True:
  codes=status(phase)
  if all(c==0 for c in codes.values()):break
  time.sleep(P['budget']['watch_interval_seconds'])
 for role,_ in assignments:assert json.loads((R/role/'EXIT.json').read_text())['exit_code']==0

try:
 assert not (R/'LAUNCHED.json').exists();assert json.loads((R/'CPU_CHECK.json').read_text())['passed']
 write('LAUNCHED.json',{'supervisor':identity(os.getpid()),'original_start':B['started_epoch'],'at':time.time()})
 run_phase('learning_development',[('fresh',3),('resume',4),('controls',5)])
 dev={}
 for role in ['fresh','resume','controls']:dev.update(json.loads((R/role/'RESULT.json').read_text()))
 selected=max(['resume','fresh'],key=lambda k:dev[k]['primary_score']);control=max(['B','F'],key=lambda k:dev[k]['primary_score'])
 assert not (R/'FROZEN_SELECTION.json').exists()
 write('FROZEN_SELECTION.json',{'at':time.time(),'selected_arm':selected,'primary_control':control,'development':dev,'locked_before_confirmation':True,'selection_rule':'frozen protocol; development only'})
 run_phase('confirmation',[('confirm_B',3),('confirm_F',4),('confirm_H',5),('confirm_N',1)])
 import sys
 sys.path.insert(0,str(R/'code'))
 from analyze import analyze
 analyze(R)
 write('SUPERVISOR_EXIT.json',{'ended':time.time(),'returncodes':{r:p.poll() for r,p in procs.items()},'elapsed_seconds':time.time()-B['started_epoch'],'bytes':storage(),'success':True})
except BaseException:
 write('HARD_STOP.json',{'at':time.time(),'error':traceback.format_exc()});stop();write('SUPERVISOR_EXIT.json',{'ended':time.time(),'failed':True,'returncodes':{r:p.poll() for r,p in procs.items()},'original_start':B['started_epoch']});raise
