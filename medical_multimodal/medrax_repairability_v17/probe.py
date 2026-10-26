"""Bounded six-call source-only repairability diagnostic using existing transport."""
from pathlib import Path
import os,json,time,sys,signal,subprocess,base64,threading
root=Path(os.environ['R1031_ROOT']);sys.path.insert(0,os.environ['R1031_TRANSPORT']);import subscription
budget=json.loads((root/'budget.json').read_text());jobs=json.loads((root/'private/jobs.json').read_text());done=threading.Event()
def save(n,x):
 p=root/n;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def own_bytes():return sum(p.stat().st_blocks*512 for p in root.rglob('*') if p.is_file() and not p.is_symlink())
def stop(reason):
 save('STOPPED.json',{'time':time.time(),'reason':reason,'original_start':budget['started_epoch'],'deadline':budget['hard_deadline_epoch']})
 p=root/'private/active_cli.json'
 if p.exists():
  x=json.loads(p.read_text());q=subprocess.run(['ps','-p',str(x['pid']),'-o','command='],text=True,capture_output=True).stdout
  if '/usr/bin/sandbox-exec' in q and '/tmp/r983-' in q:
   try:os.killpg(x['pid'],signal.SIGTERM)
   except ProcessLookupError:pass
 os._exit(124)
def watch():
 while not done.wait(3):
  try:
   size=own_bytes();save('RUN_GUARD.json',{'time':time.time(),'allocated_new_bytes':size,'deadline':budget['hard_deadline_epoch']})
   if size>=budget['own_stop_new_bytes']:stop('storage_limit')
   if time.time()>=budget['hard_deadline_epoch']:stop('original_deadline')
  except Exception:stop('guard_read_error')
assert len(jobs)==budget['max_subscription_calls']==6 and json.loads((root/'private/call_count.json').read_text())==0
save('STARTED.json',{'time':time.time(),'original_start':budget['started_epoch'],'deadline':budget['hard_deadline_epoch'],'pid':os.getpid()});threading.Thread(target=watch,daemon=True).start();fatal=None;responses=[]
system='You perform one bounded research review of supplied images and a multiple-choice question. Use only supplied current-case material. Distinguish visible observations from causal/diagnostic inference. Do not access files, execute tools, or search. Return tool_calls as an empty list and content as a JSON object with exactly these keys: answer (one uppercase letter A-F), observations (list of short strings), unresolved (list of short strings). No benchmark answer or independent adjudication is supplied.'
try:
 for n,j in enumerate(jobs):
  if time.time()>=budget['hard_deadline_epoch']:break
  parts=[{'type':'text','text':j['prompt']}]+[{'type':'image','index':i,'path':p} for i,p in enumerate(j['images'])]
  request={'system':system,'tools':[],'messages':[{'role':'user','content':parts}],'images':[base64.b64encode(Path(p).read_bytes()).decode() for p in j['images']]}
  response=subscription.execute(request,root);save(f'private/response_{n}.json',response);responses.append({'index':n,'alias':j['alias'],'arm':j['arm'],'status':response['status'],'seconds':response.get('seconds'),'images':len(j['images'])});save('PROGRESS.json',{'time':time.time(),'completed_requests':len(responses),'planned':6,'statuses':[x['status'] for x in responses]});print(json.dumps(responses[-1]),flush=True)
except Exception as e:fatal=repr(e)
finally:
 done.set();save('FINISHED.json',{'time':time.time(),'complete':len(responses)==6 and fatal is None,'planned':6,'completed_requests':len(responses),'fatal':fatal,'seconds_since_original_start':time.time()-budget['started_epoch'],'calls':responses,'new_allocated_bytes':own_bytes()})
