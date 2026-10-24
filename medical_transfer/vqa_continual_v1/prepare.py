"""Freeze group membership without using answers or model outputs."""
import collections,json,os,random,time,sys
from pathlib import Path
R=Path(os.environ['RUN_ROOT']);P=Path(os.environ['PREPARED_DATA']);C=P.parent/'campaign_12h_20261022'
sys.path.insert(0,str(R/'code'))
plan=json.loads((R/'protocol.json').read_text())
assert not (R/'private_membership.json').exists()
rows=lambda p:[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def write(name,obj):
 p=R/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2));t.replace(p)
roles={x['id']:x for x in rows(P/'audit/roles.jsonl')}
excluded=set();sources=[]
def collect(obj):
 if isinstance(obj,dict):
  if obj.get('group_id'):excluded.add(obj['group_id'])
  for k in ('id','task_id'):
   if obj.get(k) in roles:excluded.add(roles[obj[k]]['group_id'])
  for v in obj.values():collect(v)
 elif isinstance(obj,list):
  for v in obj:collect(v)
files=list((C/'runs').glob('*/private*membership*.json'))
files += [C/'runs/task_scoped_memory_v1/private_learning.json',C/'runs/semantic_boundary_v1/private_screen.jsonl',P.parent/'runs/dev_diagnostic_20261022_v1/private_frozen_membership.json']
for p in files:
 assert p.is_file(),str(p)
 collect(rows(p) if p.suffix=='.jsonl' else json.loads(p.read_text()));sources.append(str(p))
pilot=P.parent/'runs/pilot_20261022_v1/prepared_data'
for p in pilot.glob('visible/*/*.jsonl'):
 if p.stem!='test':collect(rows(p));sources.append(str(p))
for task in ['visual_qa','classification','polyp_localization']:
 for x in sorted(rows(P/f'visible/{task}/validation.jsonl'),key=lambda x:x['id'])[:4]:collect(x)
# Access only pre-existing role/type/identity metadata for stratification. Answers are
# discarded here, never inspected or used by the selector, then associated after freeze.
metadata={};visible={}
for role in set(plan['roles'].values()):
 metadata[role]={x['id']:x['metadata'] for x in rows(P/f'private/visual_qa/{role}.jsonl')}
 visible[role]=rows(P/f'visible/visual_qa/{role}.jsonl')
from selection import allocate
pools={};quotas={};availability={}
for stage in plan['order']:
 for split,n in plan['counts_per_stage'].items():
  role=plan['roles'][split]
  for qtype in plan['stages'][stage]:
   key=(stage,split,qtype);pool=collections.defaultdict(list)
   for v in visible[role]:
    m=metadata[role][v['id']]
    if m['question_type']==qtype and m['group_id'] not in excluded:
     pool[m['group_id']].append({'id':v['id'],'group_id':m['group_id'],'pixel_id':m['pixel_id'],'stage':stage,'split':split,'source_role':role,'question_type':qtype})
   pools[key]=pool;quotas[key]=n//len(plan['stages'][stage]);availability['/'.join(key)]=len(pool)
allocated=allocate(pools,quotas,plan['seed']);membership=[]
for stage in plan['order']:
 for split in plan['counts_per_stage']:
  buckets=[allocated[(stage,split,q)] for q in plan['stages'][stage]]
  membership.extend(x for pair in zip(*buckets) for x in pair)
expected=sum(plan['counts_per_stage'].values())*3
assert len(membership)==expected and len({x['group_id'] for x in membership})==expected
assert len({x['pixel_id'] for x in membership})==expected
assert not {x['group_id'] for x in membership}&excluded
write('private_membership.json',{'seed':plan['seed'],'frozen_at':time.time(),'cases':membership,'excluded_group_count':len(excluded),'exclusion_sources':sources,'answers_used_for_selection':False})
# Membership exists durably before targets are associated.
assert (R/'private_membership.json').is_file()
labels={role:{x['id']:x for x in rows(P/f'private/visual_qa/{role}.jsonl')} for role in visible}
vmap={role:{x['id']:x for x in vs} for role,vs in visible.items()}
data={s:{k:[] for k in plan['counts_per_stage']} for s in plan['order']}
for m in membership:
 role=m['source_role'];key=m['id'];v=vmap[role][key];lab=labels[role][key]
 e={**v,'task_id':key,'task_name':'visual_qa','task_type':'medical_visual_qa','split':role,'answer':lab['answer'],'metadata':{**lab['metadata'],'stream_stage':m['stage']}}
 assert Path(e['image']).is_file()
 data[m['stage']][m['split']].append(e)
write('private_data.json',data)
write('DATA_RECEIPT.json',{'passed':True,'groups':expected,'counts_per_stage':plan['counts_per_stage'],'excluded_group_count':len(excluded),'availability_before_allocation':availability,'roles_disjoint':True,'stage_groups_disjoint':True,'pixel_ids_disjoint':True,'test_opened':False,'model_calls':0,'frozen_before_target_association':True})
print(json.dumps({'groups':expected,'excluded_groups':len(excluded),'model_calls':0}))
