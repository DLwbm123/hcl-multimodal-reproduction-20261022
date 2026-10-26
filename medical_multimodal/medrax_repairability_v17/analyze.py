from pathlib import Path
import json,time,re,subprocess
from collections import Counter
b=Path('/Users/bominwang/Documents/Codex/2026-10-22/zai');r=b/'work/medrax_repairability_v17';pub=b/'work/hcl-public/medical_multimodal/medrax_repairability_v17'
assert (r/'FINISHED.json').exists(),'Wait for full finite diagnostic before reading outputs'
f=json.loads((r/'FINISHED.json').read_text());jobs=json.loads((r/'private/jobs.json').read_text());catalog=json.loads((r/'private/catalog.json').read_text());rows=[];usage=Counter();statuses=Counter();seconds=0
for n,j in enumerate(jobs):
 p=r/f'private/response_{n}.json'
 if not p.exists():rows.append({'alias':j['alias'],'arm':j['arm'],'status':'not_started','valid':False});continue
 x=json.loads(p.read_text());statuses[x['status']]+=1;seconds+=x.get('seconds',0)
 for e in x.get('events',[]):
  if e.get('type')=='turn.completed':usage.update(e['usage'])
 parsed=None;reason=None
 try:
  assert x['status']=='ok' and not x['output']['tool_calls']
  text=x['output']['content'].strip()
  if text.startswith('```'):text=re.sub(r'^```(?:json)?\s*|\s*```$','',text)
  parsed=json.loads(text);assert set(parsed)=={'answer','observations','unresolved'} and parsed['answer'] in 'ABCDEF' and len(parsed['answer'])==1 and isinstance(parsed['observations'],list) and isinstance(parsed['unresolved'],list)
 except Exception as e:reason=repr(e)
 rows.append({'alias':j['alias'],'catalog_index':j['catalog_index'],'arm':j['arm'],'status':x['status'],'valid':reason is None,'parse_error':reason,'content':parsed,'image_count':x.get('image_count'),'correct':None if reason else parsed['answer']==catalog[j['catalog_index']]['metadata']['answer'],'visible_429':'429 Too Many Requests' in x.get('stderr','')})
(r/'private/parsed_results.json').write_text(json.dumps(rows,indent=2)+'\n')
pairs=[]
for a in ['C03','C11','C13']:
 q={x['arm']:x for x in rows if x['alias']==a};ok=all(q[z]['valid'] for z in ['review','operation']);pairs.append({'alias':a,'valid_pair':ok,'review_correct':q['review'].get('correct'),'operation_correct':q['operation'].get('correct'),'benchmark_gain':bool(ok and not q['review']['correct'] and q['operation']['correct']),'benchmark_loss':bool(ok and q['review']['correct'] and not q['operation']['correct']),'clinically_verified_tool_repair':None,'independent_transfer_established':False})
agg={'time':time.time(),'complete':f['complete'],'planned_controller_calls':6,'completed_controller_calls':f['completed_requests'],'valid_structured_responses':sum(x['valid'] for x in rows),'pairs':pairs,'review_correct':sum(x.get('correct',False) is True for x in rows if x['arm']=='review'),'operation_correct':sum(x.get('correct',False) is True for x in rows if x['arm']=='operation'),'paired_benchmark_gains':sum(x['benchmark_gain'] for x in pairs),'paired_benchmark_losses':sum(x['benchmark_loss'] for x in pairs),'medical_skill_validity_established':False,'new_cases':0,'confirmation_used':0,'usage':dict(usage),'request_seconds':seconds,'visible_429_responses':sum(x.get('visible_429',False) for x in rows),'seconds_since_original_start':f['seconds_since_original_start'],'new_allocated_bytes_at_worker_exit':f['new_allocated_bytes']}
(r/'AGGREGATE_REPORT.json').write_text(json.dumps(agg,indent=2)+'\n');print(json.dumps(agg))
