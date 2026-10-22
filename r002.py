"""One fresh guarded replay; no automatic retries or subsequent experiments."""
import json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
R=Path(os.environ['REPRO_ROOT']).resolve();C=Path(__file__).resolve().parent
arms=['raw_zero_shot','static_harness','stability_hcl','plasticity_hcl']
gpus=json.loads(os.environ.get('GPU_IDS','[0,1,2,3]'))
assert len(gpus)==4 and len(set(gpus))==4
budget=json.loads((R/'budget.json').read_text());children={}
receipt=R/'replay_processes.json'
with receipt.open('x') as f:json.dump({'supervisor':os.getpid(),'started':time.time(),'fail_closed':os.environ.get('FAIL_CLOSED')=='1'},f)
def usage(nondata=False):
 args=['du','-s','-B1']
 if nondata:args+=['--exclude=raw','--exclude=images','--exclude=local_image_disjoint','--exclude=native_pixels_50_10_100']
 x=subprocess.run(args+[str(R)],capture_output=True,text=True);errors=x.stderr.splitlines()
 if not x.stdout or (x.returncode and (not errors or not all('No such file or directory' in v for v in errors))):raise RuntimeError('Storage accounting failed: '+x.stderr[:300])
 return sum(int(v.split()[0]) for v in x.stdout.splitlines())+1000000000

def stop():
 for p in children.values():
  if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
 time.sleep(15)
 for p in children.values():
  if p.poll() is None:os.killpg(p.pid,signal.SIGKILL)

def check():
 total=usage();nondata=usage(True);elapsed=time.time()-budget['started_epoch']
 (R/'replay_resource_status.json').write_text(json.dumps({'total_conservative_bytes':total,'nondata_conservative_bytes':nondata,'elapsed_seconds':elapsed,'statuses':{k:p.poll() for k,p in children.items()}},indent=2))
 if elapsed>=min(84600,budget['stop_after_seconds']) or total>=47000000000 or nondata>=39000000000:raise RuntimeError('Original time or storage guard reached')
try:
 check()
 free=[int(v) for v in subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()]
 assert all(free[g]>=60000 for g in gpus), 'Insufficient GPU headroom; no worker launched'
 for arm,gpu in zip(arms,gpus):
  cfgpath=C/f'configs/runtime_{arm}.json';cfg=json.loads(cfgpath.read_text());out=Path(cfg['storage_dir']);out.mkdir(parents=True,exist_ok=False)
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),RUN_CONFIG=str(cfgpath))
  children[arm]=subprocess.Popen([sys.executable,'-u','r001.py'],cwd=C,env=env,stdout=(out/'runner.log').open('a'),stderr=subprocess.STDOUT,start_new_session=True)
 receipt.write_text(json.dumps({'supervisor':os.getpid(),'workers':{k:p.pid for k,p in children.items()},'original_started_epoch':budget['started_epoch']},indent=2))
 while any(p.poll() is None for p in children.values()):
  check();time.sleep(15)
 (R/'replay_exit.json').write_text(json.dumps({k:p.poll() for k,p in children.items()},indent=2))
except BaseException:
 (R/'replay_failure.txt').write_text(traceback.format_exc());stop();raise
