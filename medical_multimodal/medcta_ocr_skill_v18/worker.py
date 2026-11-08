import os,sys,json,time,base64,subprocess,tempfile,shutil,re,threading,traceback
from pathlib import Path
root=Path(os.environ['R18_ROOT']);sys.path.insert(0,str(root/'code'));import subscription
budget=json.loads((root/'budget.json').read_text());jobs=json.loads((root/'private/jobs.json').read_text());source=json.loads((root/'private/source.json').read_text());start=time.time();done=threading.Event();results=[];fatal=None

def save(n,v):
 p=root/n;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2));t.replace(p)
def size():return sum(p.stat().st_blocks*512 for p in root.rglob('*') if p.is_file() and not p.is_symlink())
def watch():
 while not done.wait(5):
  try:
   n=size();save('RUN_GUARD.json',{'time':time.time(),'new_allocated_bytes':n,'limit':budget['own_stop_new_bytes']})
   if time.time()>=budget['hard_deadline_epoch'] or n>=budget['own_stop_new_bytes']:
    save('STOPPED.json',{'time':time.time(),'reason':'deadline_or_storage'});os.kill(os.getpid(),15);return
  except Exception:os.kill(os.getpid(),15);return

def call(system,messages,images,tools):
 encoded=[base64.b64encode(Path(p).read_bytes()).decode() for p in images]
 req={'system':system,'messages':messages,'tools':tools,'images':encoded}
 return subscription.execute(req,root)

def invoke(item,arm,policy,index):
 image=str(root/'private'/item['image_name']);part=[{'type':'text','text':item['question']},{'type':'image','index':0,'path':image}]
 msgs=[{'role':'user','content':part}]
 system=('Answer the user question using the current image. You may call read_ocr at most once to inspect text in the current image. '
 'OCR is noisy and is not medical ground truth. Do not use other tools or invent evidence. '
 'Return a concise direct final answer in content and an empty tool_calls list when finished. '
 'When calling read_ocr use name read_ocr with arguments_json {}. Do not simulate a tool result.')
 if arm=='source_policy':system+='\nGeneral policy distilled from prior reference trajectories, not facts about the current case:\n'+policy
 spec=[{'name':'read_ocr','description':'Extract visible text from the current image; no arguments. Output may contain OCR errors.','inputs':{},'outputs':{'text':'string'}}]
 info={'index':index,'arm':arm,'calls':0,'ocr_called':False,'ocr_completed':False,'status':'failed','answer':None}
 for turn in range(2):
  if time.time()>=budget['hard_deadline_epoch']:break
  response=call(system,msgs,[image],spec);info['calls']+=1;save('private/response_%02d_%d.json'%(index,turn),response)
  if response.get('status')!='ok':info['error']='transport_'+str(response.get('status'));break
  out=response['output'];calls=out.get('tool_calls',[])
  if not calls:
   info['answer']=out['content'].strip();info['status']='completed' if info['answer'] else 'empty';break
  if turn or len(calls)!=1 or calls[0]['name']!='read_ocr' or json.loads(calls[0]['arguments_json'])!={}:info['error']='invalid_or_excess_tool_call';break
  info['ocr_called']=True
  with tempfile.TemporaryDirectory(prefix='r18-',dir='/tmp') as td:
   dest=Path(td)/'input.jpg';shutil.copyfile(image,dest)
   try:
    o=subprocess.run(['/opt/homebrew/bin/tesseract',str(dest),'stdout'],capture_output=True,text=True,timeout=30)
    txt=o.stdout[:5000];info['ocr_completed']=o.returncode==0
   except subprocess.TimeoutExpired:txt='OCR tool timed out';info['ocr_completed']=False
  msgs.append({'role':'assistant','content':out['content'],'tool_calls':calls})
  msgs.append({'role':'tool','name':'read_ocr','content':txt if info['ocr_completed'] else 'OCR unavailable; proceed without assuming its content.'})
 return info

def norm(x):return re.sub(r'[^\w]+',' ',str(x).casefold()).strip()
def flatten(x):
 if isinstance(x,str):return [x]
 if isinstance(x,list):return sum((flatten(y) for y in x),[])
 return []

save('STARTED.json',{'time':time.time(),'pid':os.getpid(),'planned_cases':6,'planned_arms':12,'budget_calls':25});threading.Thread(target=watch,daemon=True).start()
try:
 demos=json.dumps(source,ensure_ascii=False)[:16000]
 req=[{'role':'user','content':[{'type':'text','text':'Extract a concise, general-purpose policy for deciding when and how to use OCR from these prior reference trajectories. No patient-specific findings, labels, answers, names, or cases. At most 1200 characters. Reference material: '+demos}]}]
 ans=call('Produce only a general OCR use policy in content; tool_calls must be empty.',req,[],[]);save('private/policy_generation.json',ans)
 if ans.get('status')!='ok' or ans['output'].get('tool_calls'):raise RuntimeError('policy_generation_failed')
 policy=ans['output']['content'].strip()
 if not (20<=len(policy)<=1200):raise RuntimeError('policy_length_invalid')
 save('private/policy.json',{'content':policy,'length':len(policy)});save('POLICY_FROZEN.json',{'time':time.time(),'source_count':4,'length':len(policy),'before_development':True})
 for i,item in enumerate(jobs):
  for arm in (['no_experience','source_policy'] if i%2==0 else ['source_policy','no_experience']):
   r=invoke(item,arm,policy,len(results));r['id']=item['id'];results.append(r);save('private/results_partial.json',results)
   save('PROGRESS.json',{'time':time.time(),'completed_cells':len(results),'planned_cells':12,'last_status':r['status'],'calls_used':json.loads((root/'private/call_count.json').read_text())})
   if r['status'].startswith('failed') or r['status']=='empty':raise RuntimeError('cell_failed_no_retry')
 if len(results)==12:
  gold=json.loads((root/'private/sealed_gold.json').read_text())
  for r in results:
   g=gold[r['id']];expected=[norm(x) for x in flatten(g['whitelist']) if norm(x)];black=[norm(x) for x in flatten(g['blacklist']) if norm(x)];a=norm(r['answer']);r['strict_exact']=bool(a and a in expected and not any(x==a for x in black))
  by={x['id']:{} for x in results}
  for x in results:by[x['id']][x['arm']]=x
  gain=sum(not x['no_experience']['strict_exact'] and x['source_policy']['strict_exact'] for x in by.values());loss=sum(x['no_experience']['strict_exact'] and not x['source_policy']['strict_exact'] for x in by.values())
  summary={'completed':True,'cases':6,'cells':12,'strict_exact_control':sum(x['no_experience']['strict_exact'] for x in by.values()),'strict_exact_policy':sum(x['source_policy']['strict_exact'] for x in by.values()),'gain':gain,'loss':loss,'strict_metric_note':'Exact whitelist text only: undercounts semantically correct answers; not clinical ground truth or independent patient confirmation.','ocr_calls':{a:sum(x['ocr_called'] for x in results if x['arm']==a) for a in ['no_experience','source_policy']},'calls_used':json.loads((root/'private/call_count.json').read_text()),'validation_used':False,'confirmation_used':False,'patient_independence':'unknown','causal_qualification':'development only; learned policy not matched in token length; baseline and candidate share tools and max calls'}
  save('private/scored_results.json',results);save('AGGREGATE_REPORT.json',summary)
except BaseException as e:
 fatal={'type':type(e).__name__,'message':str(e)[:300]};save('FAILED.json',{'time':time.time(),'error':fatal,'completed_cells':len(results),'planned_cells':12})
finally:
 done.set();save('FINISHED.json',{'time':time.time(),'complete':len(results)==12 and fatal is None,'fatal':fatal,'completed_cells':len(results),'planned_cells':12,'calls_used':json.loads((root/'private/call_count.json').read_text()),'new_allocated_bytes':size(),'elapsed_seconds':time.time()-budget['started_epoch']})
