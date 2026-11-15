"""Reuse frozen training roles and allocate a genuinely unused confirmation panel."""
import os,sys,json,collections
from pathlib import Path
R=Path(os.environ['RUN_ROOT']);P=json.loads((R/'protocol.json').read_text());H=R.parent;D=Path(P['prepared_data']);old=Path(P['previous_run'])
sys.path.insert(0,str(H/'vqa_continual_v1/code'))
from selection import allocate
assert not (R/'private_membership.json').exists()
probe=R/'.probe';probe.write_text('write-read');assert probe.read_text()=='write-read';probe.unlink()
roles={x['id']:x for x in map(json.loads,(D/'audit/roles.jsonl').read_text().splitlines())};groups=set();pixels=set()
def collect(obj):
 if isinstance(obj,dict):
  if obj.get('group_id'):groups.add(obj['group_id'])
  if obj.get('pixel_id'):pixels.add(obj['pixel_id'])
  for k in ['id','task_id','internal_sample_id']:
   if isinstance(obj.get(k),str) and obj[k] in roles:
    x=roles[obj[k]];groups.add(x['group_id']);pixels.add(x['pixel_id'])
  for v in obj.values():collect(v)
 elif isinstance(obj,list):
  for x in obj:collect(x)
prev=json.loads((old/'private_membership.json').read_text())
for s in prev['exclusion_sources']:
 p=Path(s);collect([json.loads(x) for x in p.read_text().splitlines()] if p.suffix=='.jsonl' else json.loads(p.read_text()))
collect(prev);data=json.loads((old/'private_learning.json').read_text());collect(data);collect(json.loads(Path(P['old_state']).read_text())['memory'])
for task in ['visual_qa','classification','polyp_localization']:
 for x in sorted(map(json.loads,(D/f'visible/{task}/validation.jsonl').read_text().splitlines()),key=lambda x:x['id'])[:4]:collect(x)
types=['yes_no_or_presence','count','color_set','instrument','location_set','choice_or_other'];meta={x['id']:x['metadata'] for x in map(json.loads,(D/'private/visual_qa/h_search.jsonl').read_text().splitlines())}
pools={t:collections.defaultdict(list) for t in types};quotas={t:60 if t==types[0] else 28 for t in types}
for x in map(json.loads,(D/'visible/visual_qa/h_search.jsonl').read_text().splitlines()):
 m=meta[x['id']]
 if m['group_id'] not in groups and m['pixel_id'] not in pixels:pools[m['question_type']][m['group_id']].append({**x,'metadata':m})
chosen=allocate(pools,quotas,P['seed']);members={k:v for k,v in prev['members'].items() if not k.startswith('confirmation:')}
for t,rows in chosen.items():members['confirmation:'+t]=[{'id':x['id'],'group_id':x['metadata']['group_id'],'pixel_id':x['metadata']['pixel_id']} for x in rows]
new=[x for t in types for x in members['confirmation:'+t]];flat=[x for v in members.values() for x in v]
assert len(new)==200 and len(flat)==376==len({x['group_id'] for x in flat})==len({x['pixel_id'] for x in flat})
assert not {x['group_id'] for x in new}&groups and not {x['pixel_id'] for x in new}&pixels
(R/'private_membership.json').write_text(json.dumps({'seed':P['seed'],'members':members,'excluded_groups':len(groups),'training_source':str(old),'exclusion_sources':prev['exclusion_sources']+[str(old/'private_membership.json')]},indent=2))
labels={x['id']:x['answer'] for x in map(json.loads,(D/'private/visual_qa/h_search.jsonl').read_text().splitlines())};examples=[]
for i in range(60):
 for t in types:
  if i>=len(chosen[t]):continue
  x=chosen[t][i];m=dict(x['metadata']);m.update(original_role=m['role'],study_role='confirmation',role='validation');answer=labels[x['id']];assert isinstance(answer,str) and answer.strip()
  examples.append({**x,'metadata':m,'answer':answer,'task_id':x['id'],'task_name':'visual_qa','task_type':'medical_visual_qa','split':'confirmation'})
(R/'private_confirmation.json').write_text(json.dumps(examples));(R/'private_learning.json').write_text(json.dumps(data))
receipt={'counts':{'train':48,'selection':48,'development':80,'confirmation':200},'training_reused_exactly':True,'new_confirmation_no_prior_group_pixel_overlap':True,'original_test_untouched':True,'patient_independence':'unverified','model_calls':0,'excluded_groups':len(groups)}
(R/'DATA_CHECK.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
