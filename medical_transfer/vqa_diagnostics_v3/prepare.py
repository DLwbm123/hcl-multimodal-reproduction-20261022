"""Reserve development review groups without consulting their answers or predictions."""
import json,os,sys,time,collections,copy
from pathlib import Path
R=Path(os.environ['RUN_ROOT']);P=Path(os.environ['PREPARED_DATA']);V=R.parent/'vqa_local_v2';V1=R.parent/'vqa_continual_v1'
sys.path.insert(0,str(R/'code'))
from selection import allocate
plan=json.loads((R/'protocol.json').read_text());assert not (R/'private_membership.json').exists()
rows=lambda p:[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
roles={x['id']:x for x in rows(P/'audit/roles.jsonl')};excluded=set();pixels=set()
def collect(obj):
 if isinstance(obj,dict):
  if obj.get('group_id'):excluded.add(obj['group_id'])
  if obj.get('pixel_id'):pixels.add(obj['pixel_id'])
  for k in ['id','task_id']:
   if obj.get(k) in roles:
    rr=roles[obj[k]];excluded.add(rr['group_id'])
    if rr.get('pixel_id'):pixels.add(rr['pixel_id'])
  for v in obj.values():collect(v)
 elif isinstance(obj,list):
  for v in obj:collect(v)
m1=json.loads((V1/'private_membership.json').read_text())
for f in m1['exclusion_sources']:
 p=Path(f);collect(rows(p) if p.suffix=='.jsonl' else json.loads(p.read_text()))
for task in ['visual_qa','classification','polyp_localization']:
 for x in sorted(rows(P/f'visible/{task}/validation.jsonl'),key=lambda x:x['id'])[:4]:collect(x)
collect(m1);collect(json.loads((V/'private_membership.json').read_text()))
meta={x['id']:x['metadata'] for x in rows(P/'private/visual_qa/h_search.jsonl')};vis=rows(P/'visible/visual_qa/h_search.jsonl');pools={};quotas={}
for stage,types in plan['stages'].items():
 for q in types:
  pool=collections.defaultdict(list)
  for v in vis:
   m=meta[v['id']]
   if m['question_type']==q and m['group_id'] not in excluded and m['pixel_id'] not in pixels:
    pool[m['group_id']].append({'id':v['id'],'group_id':m['group_id'],'pixel_id':m['pixel_id'],'stage':stage,'role':'dev_review','source_role':'h_search','question_type':q})
  pools[stage,q]=pool;quotas[stage,q]=12
allocated=allocate(pools,quotas,plan['seed']);fresh=[]
for st in plan['order']:
 a,b=[allocated[st,q] for q in plan['stages'][st]];fresh.extend(x for pair in zip(a,b) for x in pair)
assert len(fresh)==len({x['group_id'] for x in fresh})==len({x['pixel_id'] for x in fresh})==72
(R/'private_membership.json').write_text(json.dumps({'cases':fresh,'frozen_at':time.time(),'excluded_groups':len(excluded),'answers_used':False,'reused_development_source':'vqa_local_v2 private_data roles train/selection/anchor only'},indent=2))
d=json.loads((V/'private_data.json').read_text());data={st:{k:copy.deepcopy(es) for k,es in blocks.items() if k!='report'} for st,blocks in d.items()}
labels={x['id']:x for x in rows(P/'private/visual_qa/h_search.jsonl')};vm={x['id']:x for x in vis}
for st in plan['order']:data[st]['review']=[]
for m in fresh:
 v=vm[m['id']];lab=labels[m['id']]
 data[m['stage']]['review'].append({**v,'task_id':m['id'],'task_name':'visual_qa','task_type':'medical_visual_qa','split':'dev_review','answer':lab['answer'],'metadata':{**lab['metadata'],'original_role':'h_search','role':'dev_review','stream_stage':m['stage']}})
(R/'private_data.json').write_text(json.dumps(data))
for name,arm,stage in [('after_A','bulk','A'),('after_B','stability','B')]:
 state=json.loads((V/arm/f'private_committed_{arm}_{stage}.json').read_text())['memory'];(R/f'private_{name}.json').write_text(json.dumps(state))
def flagged(e):return str(e['question']).strip().lower() in {'','nan','null','none'} or str(e['answer']).strip().lower() in {'','nan','null'}
quality={st:{role:{'n':len(es),'missing_field_flags':sum(flagged(e) for e in es)} for role,es in blocks.items()} for st,blocks in data.items()}
receipt={'passed':True,'fresh_review_groups':72,'reused_development_groups':288,'excluded_history_groups':len(excluded),'all_unique_groups_pixels':True,'no_report_or_test_used':True,'target_association_after_membership':True,'quality':quality,'model_calls':0}
(R/'DATA_RECEIPT.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
