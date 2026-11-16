"""Independent hard deadline and storage watchdog; only own child group is terminated."""
import os,json,time,signal,subprocess,traceback,resource
from pathlib import Path
R=Path(os.environ['RUN_ROOT']);P=json.loads((R/'protocol.json').read_text());B=json.loads((R/'budget.json').read_text());child=None;identity=None

def write(n,x):
 p=R/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def ident(pid):
 p=Path('/proc')/str(pid)
 return {'pid':pid,'start_ticks':int((p/'stat').read_text().rsplit(')',1)[1].split()[19]),'pgid':os.getpgid(pid),'argv':(p/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip()}
def storage():
 total=0
 for directory,dirs,files in os.walk(R):
  for f in files:
   try:total+=os.stat(os.path.join(directory,f),follow_symlinks=False).st_size
   except FileNotFoundError:
    if not f.endswith('.tmp'):raise
 return total

def stop():
 if child is not None and child.poll() is None:
  try:matching=ident(child.pid)==identity
  except (FileNotFoundError,ProcessLookupError):matching=False
  if matching and identity['pgid']==child.pid:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
def alarm(*_):raise RuntimeError('hard deadline or external termination')
signal.signal(signal.SIGALRM,alarm);signal.signal(signal.SIGTERM,alarm);signal.setitimer(signal.ITIMER_REAL,max(.1,B['hard_deadline_epoch']-time.time()))
resource.setrlimit(resource.RLIMIT_FSIZE,(512*1024**2,512*1024**2))
try:
 assert json.loads((R/'CPU_CHECK.json').read_text())['passed']
 assert not (R/'LAUNCHED.json').exists()
 size=storage();write('resource_status.json',{'time':time.time(),'bytes':size})
 free=int(subprocess.check_output(['nvidia-smi',f'--id={P["budget"]["gpu"]}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>=P['budget']['min_gpu_free_mib']
 child=subprocess.Popen([os.environ['WORKER_PYTHON'],os.environ['WORKER_ENTRY']],cwd=R,env={**os.environ,'CUDA_VISIBLE_DEVICES':str(P['budget']['gpu'])},stdout=(R/'runner.log').open('a'),stderr=subprocess.STDOUT,start_new_session=True)
 identity=ident(child.pid)
 assert all(t not in identity['argv'].lower() for t in [x for x in os.environ.get('PRIVATE_PROCESS_TERMS','hcl,skill,polyp').split(',') if x])
 write('LAUNCHED.json',{'supervisor':ident(os.getpid()),'worker':identity,'started':B['started_epoch'],'gpu':P['budget']['gpu']})
 while child.poll() is None:
  size=storage();write('resource_status.json',{'time':time.time(),'bytes':size,'returncode':child.poll()})
  if size>=P['budget']['storage_stop_bytes']:raise RuntimeError('storage early stop')
  if time.time()>=B['soft_deadline_epoch']:raise RuntimeError('original soft deadline')
  time.sleep(P['budget']['watch_seconds'])
 if child.returncode!=0:raise RuntimeError('worker failed')
 write('SUPERVISOR_EXIT.json',{'success':True,'returncode':child.returncode,'ended':time.time(),'elapsed_seconds':time.time()-B['started_epoch'],'bytes':storage()})
except BaseException:
 write('HARD_STOP.json',{'at':time.time(),'error':traceback.format_exc()});stop();write('SUPERVISOR_EXIT.json',{'success':False,'ended':time.time(),'returncode':child.poll() if child else None});raise
