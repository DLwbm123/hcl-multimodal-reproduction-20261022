import os,sys,json,time,subprocess,signal,traceback
from pathlib import Path
C=Path(os.environ['CAMPAIGN_ROOT']);R=C/'runs/semantic_boundary_v1';B=json.loads((R/'budget.json').read_text())
sys.path.insert(0,str(C/'code'))
from guard import identity
record={**identity(os.getpid()),'mode':'pid'};f=C/'registry'/f'{os.getpid()}.tmp';f.write_text(json.dumps(record));f.replace(f.with_suffix('.json'))
proc=None
try:
 if (R/'EXIT.json').exists():raise RuntimeError('no implicit restart')
 status=json.loads((C/'campaign_status.json').read_text())
 if (C/'CAMPAIGN_STOP.json').exists() or time.time()-status['time']>120 or time.time()>=B['early_stop_epoch']:raise RuntimeError('campaign not available')
 free=int(subprocess.check_output(['nvidia-smi','--id=4','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
 if free<35000:raise RuntimeError('GPU4 headroom insufficient')
 log=(R/'runner.log').open('a');env={**os.environ,'CUDA_VISIBLE_DEVICES':'4','PYTHONDONTWRITEBYTECODE':'1'}
 proc=subprocess.Popen([os.environ['MODEL_PYTHON'],os.environ['NEUTRAL_WORKER']],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 (R/'processes.json').write_text(json.dumps({'supervisor':os.getpid(),'worker':proc.pid,'gpu':4,'started_epoch':B['started_epoch']}))
 while proc.poll() is None:
  status=json.loads((C/'campaign_status.json').read_text())
  size=int(subprocess.check_output(['du','-sb','.'],cwd=R,text=True,timeout=30).split()[0]);count=json.loads((R/'model_attempts.json').read_text())['attempts'] if (R/'model_attempts.json').exists() else 0
  f=R/'resource_status.tmp';f.write_text(json.dumps({'time':time.time(),'bytes':size,'calls':count,'elapsed_seconds':time.time()-B['started_epoch']}));f.replace(R/'resource_status.json')
  if (C/'CAMPAIGN_STOP.json').exists() or time.time()-status['time']>120 or time.time()>=B['early_stop_epoch'] or size>=B['early_stop_bytes'] or count>B['max_calls']:raise RuntimeError('guard limit or unavailable campaign')
  time.sleep(10)
 (R/'SUPERVISOR_EXIT.json').write_text(json.dumps({'exit_code':proc.returncode,'ended':time.time()}))
except BaseException:
 (R/'HARD_STOP.json').write_text(json.dumps({'error':traceback.format_exc(),'time':time.time()}))
 if proc and proc.poll() is None:
  os.killpg(proc.pid,signal.SIGTERM)
  try:proc.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL)
 raise
