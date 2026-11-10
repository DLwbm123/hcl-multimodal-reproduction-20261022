"""Allocate unseen source groups before joining labels; never touch original test."""
import os,json,collections,sys
from pathlib import Path
R=Path(os.environ['RUN_ROOT']);H=R.parent;P=H/'medical_transfer_v1/prepared_data'
sys.path.insert(0,str(H/'vqa_continual_v1/code'))
from selection import allocate
assert not (R/'private_membership.json').exists()
R.mkdir(exist_ok=True);probe=R/'.probe';probe.write_text('write-read');assert probe.read_text()=='write-read';probe.unlink()
roles={x['id']:x for x in map(json.loads,(P/'audit/roles.jsonl').read_text().splitlines())};excluded=set();pixels=set();sources=[]
def collect(obj):
 if isinstance(obj,dict):
  if obj.get('group_id'):excluded.add(obj['group_id'])
  if obj.get('pixel_id'):pixels.add(obj['pixel_id'])
  for k in ['id','task_id','internal_sample_id']:
   if isinstance(obj.get(k),str) and obj[k] in roles:
    x=roles[obj[k]];excluded.add(x['group_id']);pixels.add(x['pixel_id'])
  for v in obj.values():collect(v)
 elif isinstance(obj,list):
  for x in obj:collect(x)
def read(p):
 assert p.is_file(),p
 obj=[json.loads(x) for x in p.read_text().splitlines()] if p.suffix=='.jsonl' else json.loads(p.read_text());collect(obj);sources.append(str(p))
for p in json.loads((H/'vqa_continual_v1/private_membership.json').read_text())['exclusion_sources']:read(Path(p))
for n in ['vqa_continual_v1','vqa_local_v2','vqa_diagnostics_v3']:
 read(H/n/'private_membership.json');read(H/n/'private_data.json')
read(H/'graph_merge_20261109/private/selected_panel.json')
for task in ['visual_qa','classification','polyp_localization']:
 for x in sorted(map(json.loads,(P/f'visible/{task}/validation.jsonl').read_text().splitlines()),key=lambda x:x['id'])[:4]:collect(x)
types=['yes_no_or_presence','count','color_set','instrument','location_set','choice_or_other'];pools={};quotas={};labels={}
for role,source_role in [('train','h_search'),('selection','h_search'),('development','h_search'),('confirmation','train_replay')]:
 meta={x['id']:x['metadata'] for x in map(json.loads,(P/f'private/visual_qa/{source_role}.jsonl').read_text().splitlines())}
 for t in types:
  key=role+':'+t;pools[key]=collections.defaultdict(list)
  quotas[key]=(60 if t==types[0] else 28) if role=='confirmation' else (14 if t in types[:2] else 13) if role=='development' else 8
 for x in map(json.loads,(P/f'visible/visual_qa/{source_role}.jsonl').read_text().splitlines()):
  m=meta[x['id']]
  if m['group_id'] in excluded or m['pixel_id'] in pixels:continue
  pools[role+':'+m['question_type']][m['group_id']].append({**x,'metadata':m})
chosen=allocate(pools,quotas,20261110);membership={k:[{'id':x['id'],'group_id':x['metadata']['group_id'],'pixel_id':x['metadata']['pixel_id']} for x in v] for k,v in chosen.items()}
flat=[x for v in membership.values() for x in v]
assert len(flat)==376 and len({x['group_id'] for x in flat})==376 and len({x['pixel_id'] for x in flat})==376
assert not {x['group_id'] for x in flat}&excluded and not {x['pixel_id'] for x in flat}&pixels
(R/'private_membership.json').write_text(json.dumps({'seed':20261110,'exclusion_sources':sources,'excluded_group_count':len(excluded),'members':membership},indent=2))
# Membership is now frozen. Only now join target values into private execution files.
for source_role in ['h_search','train_replay']:
 labels.update({x['id']:x['answer'] for x in map(json.loads,(P/f'private/visual_qa/{source_role}.jsonl').read_text().splitlines())})
data={r:[] for r in ['train','selection','development','confirmation']}
for key,rows in chosen.items():
 role,t=key.split(':')
 for x in rows:
  m=dict(x['metadata']);m['original_role']=m['role'];m['study_role']=role;m['role']='h_search' if role=='train' else 'validation'
  answer=labels[x['id']];assert isinstance(answer,str) and answer.strip(), 'invalid target; no result-dependent filtering'
  data[role].append({**x,'metadata':m,'answer':answer,'task_id':x['id'],'task_name':'visual_qa','task_type':'medical_visual_qa','split':role})
# Interleave types; both learning batches have exactly four examples of each type.
for role in data:
 buckets={t:[x for x in data[role] if x['metadata']['question_type']==t] for t in types}
 data[role]=[buckets[t][i] for i in range(max(map(len,buckets.values()))) for t in types if i<len(buckets[t])]
fixed=[next(x for x in data['train'] if x['metadata']['question_type']==t) for t in types]
(R/'private_confirmation.json').write_text(json.dumps(data.pop('confirmation')))
(R/'private_learning.json').write_text(json.dumps({**data,'fixed_examples':[{'question':x['question'],'answer':x['answer']} for x in fixed]}))
receipt={'counts':{r:sum(v for k,v in quotas.items() if k.startswith(r+':')) for r in ['train','selection','development','confirmation']},'confirmation_quotas':{t:quotas['confirmation:'+t] for t in types},'no_group_or_pixel_overlap':True,'historical_exclusions':len(excluded),'original_test_untouched':True,'confirmation_source':'unused train_replay; study-only role reassignment','patient_independence':'unverified','model_calls':0}
(R/'DATA_CHECK.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
