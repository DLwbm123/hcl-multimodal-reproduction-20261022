import os,sys,json,time,subprocess,signal,resource,traceback
from pathlib import Path
R=Path(os.environ['RUN_ROOT']);P=json.loads((R/'protocol.json').read_text());B=json.loads((R/'budget.json').read_text());child=None;record=None
resource.setrlimit(resource.RLIMIT_FSIZE,(32*1024**2,32*1024**2))
def write(n,x):
 p=R/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def identity(pid):
 p=Path('/proc')/str(pid);return {'pid':pid,'pgid':os.getpgid(pid),'ticks':int((p/'stat').read_text().rsplit(')',1)[1].split()[19]),'argv':(p/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip()}
def stop(*args):raise RuntimeError('hard deadline or termination')
signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,max(.1,B['hard_deadline_epoch']-time.time()))
def status():
 size=max(int(subprocess.check_output(a,cwd=R,text=True).split()[0]) for a in [['du','-sb','.'],['du','-s','-B1','.']])+1048576
 write('resource_status.json',{'time':time.time(),'bytes':size})
 if size>=P['storage_stop_bytes']:raise RuntimeError('storage limit')
 if time.time()>=B['soft_deadline_epoch']:raise RuntimeError('deadline')
try:
 assert not (R/'LAUNCHED.json').exists();status()
 free=int(subprocess.check_output(['nvidia-smi',f"--id={P['gpu']}",'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>35000
 child=subprocess.Popen([os.environ['WORKER_PYTHON'],os.environ['WORKER_ENTRY']],cwd=R,env={**os.environ,'CUDA_VISIBLE_DEVICES':str(P['gpu'])},stdout=(R/'runner.log').open('a'),stderr=subprocess.STDOUT,start_new_session=True);record=identity(child.pid)
 assert record['pgid']==child.pid
 assert all(t not in record['argv'].lower() for t in os.environ['FORBIDDEN_PROCESS_TERMS'].split(','))
 write('LAUNCHED.json',{'supervisor':identity(os.getpid()),'worker':record})
 while child.poll() is None:status();time.sleep(5)
 assert child.returncode==0,'worker failed';assert json.loads((R/'EXIT.json').read_text())['exit_code']==0
 write('SUPERVISOR_EXIT.json',{'success':True,'ended':time.time(),'elapsed_seconds':time.time()-B['started_epoch']})
except BaseException:
 write('HARD_STOP.json',{'error':traceback.format_exc(),'at':time.time()})
 if child is not None and child.poll() is None and identity(child.pid)==record:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
 write('SUPERVISOR_EXIT.json',{'success':False,'ended':time.time()});raise
