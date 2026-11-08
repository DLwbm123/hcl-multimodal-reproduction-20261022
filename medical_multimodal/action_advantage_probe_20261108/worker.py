import os,sys,json,time,random,re,decimal,threading
from pathlib import Path
D=decimal.Decimal
root=Path(os.environ['RUN_ROOT']);sys.path.insert(0,str(root/'code'));import subscription
budget=json.loads((root/'budget.json').read_text());done=threading.Event()
def save(name,obj):
 p=root/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2));t.replace(p)
def allocated():return sum(p.stat().st_blocks*512 for p in root.rglob('*') if p.is_file() and not p.is_symlink())
def watch():
 while not done.wait(3):
  n=allocated();save('RUN_GUARD.json',{'time':time.time(),'allocated_bytes':n})
  if n>=budget['own_stop_new_bytes'] or time.time()>=budget['hard_deadline_epoch']:os.kill(os.getpid(),15)
def generate():
 r=random.Random(2601108); result=[]
 for i in range(12):
  kind=['compute','read'][i%2];a=r.randrange(1101,9800);b=r.randrange(31,97);value=r.randrange(101,999)
  question=(f'Synthetic nonclinical instrument worksheet. Recorded A={a}, B={b}, reference={value}. '+('Calculate A × B / 1000, rounded to exactly two decimal places.' if kind=='compute' else 'Return the recorded reference value as a number with exactly two decimal places.'))
  gold=(D(a*b)/D(1000)).quantize(D('.01'),rounding=decimal.ROUND_HALF_UP) if kind=='compute' else D(value).quantize(D('.01'))
  result.append({'kind':kind,'question':question,'product':a*b,'gold':str(gold)})
 return result
def score(answer,gold):
 if not re.fullmatch(r'\s*[-+]?\d+(?:\.\d+)?\s*',answer):return False,True
 try:return D(answer.strip())==D(gold),False
 except decimal.InvalidOperation:return False,True
def query(item,action,phase,index):
 system='Return only the requested number, with exactly two decimal places, as content. The data are synthetic and nonclinical. tool_calls must be empty.'
 if action:system+='\nA local calculator returned the intermediate A × B = '+str(item['product'])+'. Use it only if relevant to the question.'
 req={'system':system,'messages':[{'role':'user','content':[{'type':'text','text':item['question']}]}],'tools':[],'images':[]}
 response=subscription.execute(req,root);save('private/response_%s_%02d.json'%(phase,index),response)
 if response.get('status')!='ok' or response['output'].get('tool_calls'):raise RuntimeError('model_transport_or_unexpected_tool')
 return response['output']['content'].strip()
items=generate();assert len(items)==12 and score('1.50','1.50')==(True,False) and score('not 1.50','1.50')==(False,True)
save('private/synthetic_cases.json',items);save('STARTED.json',{'time':time.time(),'pid':os.getpid(),'source_cells':8,'development_cells':32});threading.Thread(target=watch,daemon=True).start();start=time.time();calls=[];fatal=None
try:
 for i,item in enumerate(items[:4]):
  for action in ([False,True] if i%2==0 else [True,False]):
   answer=query(item,action,'source',len(calls));calls.append({'i':i,'action':action,'answer':answer});save('PROGRESS.json',{'phase':'source','completed_cells':len(calls),'planned_cells':40})
 policy={}
 for k in ('compute','read'):
  gain=loss=0
  for i,item in enumerate(items[:4]):
   if item['kind']!=k:continue
   rows={x['action']:score(x['answer'],item['gold'])[0] for x in calls if x['i']==i};gain+=int(not rows[False] and rows[True]);loss+=int(rows[False] and not rows[True])
  policy[k]={'gain':gain,'loss':loss,'action':gain>loss}
 save('private/frozen_policy.json',policy)
 dev=items[4:];actions=[policy[x['kind']]['action'] for x in dev];sham=actions.copy();random.Random(91827).shuffle(sham)
 rows=[];arms=['raw','fixed','learned','sham']
 for i,item in enumerate(dev):
  for arm in arms[i%4:]+arms[:i%4]:
   action={'raw':False,'fixed':True,'learned':actions[i],'sham':sham[i]}[arm]
   answer=query(item,action,'development',len(rows));rows.append({'i':i,'arm':arm,'action':action,'answer':answer});save('private/results_partial.json',rows);save('PROGRESS.json',{'phase':'development','completed_cells':8+len(rows),'planned_cells':40})
 by={i:{} for i in range(8)};invalid={a:0 for a in arms}
 for row in rows:
  correct,bad=score(row['answer'],dev[row['i']]['gold']);invalid[row['arm']]+=int(bad);by[row['i']][row['arm']]=correct
 pairs={}
 for arm in arms[1:]:
  gain=sum(not b['raw'] and b[arm] for b in by.values());loss=sum(b['raw'] and not b[arm] for b in by.values());pairs[arm]={'gain':gain,'loss':loss,'net':gain-loss}
 save('AGGREGATE_REPORT.json',{'complete':True,'source_cases':4,'development_cases':8,'development_cells':32,'source_policy':policy,'accuracy':{a:sum(b[a] for b in by.values()) for a in arms},'invalid':invalid,'paired_vs_raw':pairs,'action_counts':{a:sum(x['action'] for x in rows if x['arm']==a) for a in arms},'calls_used':json.loads((root/'private/call_count.json').read_text()),'interpretation':'Synthetic exploratory test only; no clinical performance, autonomous tool use, reinforcement learning, or continual retention established.'})
except BaseException as e:
 fatal={'type':type(e).__name__,'message':str(e)[:300]};save('FAILED.json',{'time':time.time(),'error':fatal,'calls_used':json.loads((root/'private/call_count.json').read_text())})
finally:
 done.set();save('FINISHED.json',{'time':time.time(),'complete':fatal is None,'fatal':fatal,'calls_used':json.loads((root/'private/call_count.json').read_text()),'elapsed_seconds':time.time()-budget['started_epoch'],'allocated_bytes':allocated()})
