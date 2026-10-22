import os,time,json,subprocess,signal,traceback
from pathlib import Path
ROOT=Path(os.environ['HCL_DIAGNOSTIC_ROOT']).resolve()
B=json.loads((ROOT/'budget.json').read_text());proc=None
try:
    free=int(subprocess.check_output(['nvidia-smi','--id=4','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    if free<35000:raise RuntimeError('GPU4 headroom insufficient; no worker dispatched')
    log=(ROOT/'runner.log').open('a')
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'4','PYTHONDONTWRITEBYTECODE':'1'}
    proc=subprocess.Popen([os.environ['HCL_PYTHON'],os.environ['HCL_NEUTRAL_WORKER_ENTRY']],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (ROOT/'processes.json').write_text(json.dumps({'supervisor':os.getpid(),'worker':proc.pid,'gpu':4,'started_epoch':B['started_epoch']}))
    while proc.poll() is None:
        size=int(subprocess.check_output(['du','-sb','.'],cwd=ROOT,text=True).split()[0])
        attempts=json.loads((ROOT/'model_attempts.json').read_text())['attempts'] if (ROOT/'model_attempts.json').exists() else 0
        elapsed=time.time()-B['started_epoch']
        tmp=ROOT/'resource_status.tmp';tmp.write_text(json.dumps({'time':time.time(),'elapsed_seconds':elapsed,'new_bytes':size,'model_attempts':attempts,'worker_pid':proc.pid}));tmp.replace(ROOT/'resource_status.json')
        if elapsed>=B['early_stop_seconds']:raise RuntimeError('bounded diagnostic time guard')
        if size>=B['early_stop_new_bytes']:raise RuntimeError('bounded diagnostic storage guard')
        if attempts>B['max_calls']:raise RuntimeError('bounded diagnostic call guard')
        time.sleep(10)
    (ROOT/'SUPERVISOR_EXIT.json').write_text(json.dumps({'exit_code':proc.returncode,'ended':time.time()}))
except BaseException:
    (ROOT/'HARD_STOP.json').write_text(json.dumps({'error':traceback.format_exc(),'time':time.time()}))
    if proc and proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL)
    raise
