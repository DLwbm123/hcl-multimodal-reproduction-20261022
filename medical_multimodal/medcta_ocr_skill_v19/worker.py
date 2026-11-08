import os,sys,json,time,base64,subprocess,tempfile,shutil,re,threading
from pathlib import Path
root=Path(os.environ['R18_ROOT']);sys.path.insert(0,str(root/'code'));import subscription
budget=json.loads((root/'budget.json').read_text());jobs=json.loads((root/'private/jobs.json').read_text());policy=json.loads((root/'private/policy.json').read_text())['content'];done=threading.Event();results=[];fatal=None

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

def flatten(x):
 if isinstance(x,str):return [x]
 if isinstance(x,list):return sum((flatten(y) for y in x),[])
 return []
def norm(x):return re.sub(r'[^\w]+',' ',str(x).casefold()).strip()
def contains(h,n):return bool(n and re.search(r'(?<!\w)'+re.escape(n)+r'(?!\w)',h))

save('STARTED.json',{'time':time.time(),'pid':os.getpid(),'planned_cases':6,'planned_cells':18});threading.Thread(target=watch,daemon=True).start()
try:
 for i,item in enumerate(jobs):
  image=str(root/'private'/item['image_name'])
  with tempfile.TemporaryDirectory(prefix='r19-',dir='/tmp') as td:
   dest=Path(td)/'input.jpg';shutil.copyfile(image,dest)
   o=subprocess.run(['/opt/homebrew/bin/tesseract',str(dest.resolve()),'stdout'],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=30)
  if o.returncode!=0:raise RuntimeError('ocr_failed_case_'+str(i))
  ocr=o.stdout[:5000];save('private/ocr_%02d.json'%i,{'text':ocr,'characters':len(ocr),'completed':True})
  arms=['H0','fixed_ocr','fixed_ocr_policy'];arms=arms[i%3:]+arms[:i%3]
  for arm in arms:
   system='Answer the current clinical question from the provided current image. Return a brief direct final answer in content; tool_calls must be empty. Do not invent findings.'
   if arm!='H0':system+='\nThe following is actual OCR text from the current image. It may contain errors and is not medical truth. Use it only when relevant:\n'+ocr
   if arm=='fixed_ocr_policy':system+='\nGeneral workflow distilled from prior reference cases, not facts about this case:\n'+policy
   parts=[{'type':'text','text':item['question']},{'type':'image','index':0,'path':image}]
   req={'system':system,'messages':[{'role':'user','content':parts}],'tools':[],'images':[base64.b64encode(Path(image).read_bytes()).decode()]}
   response=subscription.execute(req,root);save('private/response_%02d.json'%len(results),response)
   row={'id':item['id'],'arm':arm,'status':response.get('status'),'ocr_characters':len(ocr) if arm!='H0' else 0,'answer':None}
   if response.get('status')!='ok' or response['output'].get('tool_calls'):
    results.append(row);save('private/results_partial.json',results);raise RuntimeError('response_failed_no_retry')
   row['answer']=response['output']['content'].strip();row['status']='completed' if row['answer'] else 'empty';results.append(row)
   save('private/results_partial.json',results);save('PROGRESS.json',{'time':time.time(),'completed_cells':len(results),'planned_cells':18,'completed_cases':i,'last_status':row['status'],'calls_used':json.loads((root/'private/call_count.json').read_text())})
   if not row['answer']:raise RuntimeError('empty_response_no_retry')
 if len(results)==18:
  gold=json.loads((root/'private/sealed_gold.json').read_text())
  for r in results:
   g=gold[r['id']];ws=[norm(x) for x in flatten(g['whitelist']) if norm(x)];bs=[norm(x) for x in flatten(g['blacklist']) if norm(x)];a=norm(r['answer'])
   r['keyword_proxy']=bool(a and ws and all(contains(a,w) for w in ws) and not any(contains(a,b) for b in bs));r['strict_exact']=bool(a and a in ws and not any(a==b for b in bs))
  by={x['id']:{} for x in results}
  for x in results:by[x['id']][x['arm']]=x
  def pair(a,b):
   gain=sum(not x[a]['keyword_proxy'] and x[b]['keyword_proxy'] for x in by.values());loss=sum(x[a]['keyword_proxy'] and not x[b]['keyword_proxy'] for x in by.values());return {'gain':gain,'loss':loss,'net':gain-loss}
  save('private/scored_results.json',results)
  save('AGGREGATE_REPORT.json',{'complete':True,'cases':6,'cells':18,'keyword_proxy':{a:sum(x[a]['keyword_proxy'] for x in by.values()) for a in ['H0','fixed_ocr','fixed_ocr_policy']},'strict_exact':{a:sum(x[a]['strict_exact'] for x in by.values()) for a in ['H0','fixed_ocr','fixed_ocr_policy']},'pairs':{'H0_to_ocr':pair('H0','fixed_ocr'),'ocr_to_policy':pair('fixed_ocr','fixed_ocr_policy'),'H0_to_policy':pair('H0','fixed_ocr_policy')},'ocr_executions':6,'ocr_shared_equally':True,'remaining_validation_cases':6,'confirmation_reserved':68,'patient_independence':'unknown','calls_used':json.loads((root/'private/call_count.json').read_text()),'metric_limit':'Keyword matching does not establish medical correctness; semantically correct paraphrases may be missed and mentions/negations may yield false positives. Only development evidence.'})
except BaseException as e:
 fatal={'type':type(e).__name__,'message':str(e)[:300]};save('FAILED.json',{'time':time.time(),'error':fatal,'completed_cells':len(results),'planned_cells':18})
finally:
 done.set();save('FINISHED.json',{'time':time.time(),'complete':len(results)==18 and fatal is None,'fatal':fatal,'completed_cells':len(results),'planned_cells':18,'calls_used':json.loads((root/'private/call_count.json').read_text()),'new_allocated_bytes':size(),'elapsed_seconds':time.time()-budget['started_epoch']})
