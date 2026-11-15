"""Freeze new training, selection, development and confirmation groups before inference."""
import os,sys,json,collections
from pathlib import Path
R=Path(os.environ['RUN_ROOT']);P=json.loads((R/'protocol.json').read_text());H=R.parent;D=Path(P['prepared_data']);sys.path.insert(0,str(H/'vqa_continual_v1/code'))
from selection import allocate
assert not (R/'private_membership.json').exists()
probe=R/'.probe';probe.write_text('ok');assert probe.read_text()=='ok';probe.unlink()
roles={x['id']:x for x in map(json.loads,(D/'audit/roles.jsonl').read_text().splitlines())};groups=set();pixels=set();sources=[]
def collect(o):
 if isinstance(o,dict):
  if o.get('group_id'):groups.add(o['group_id'])
  if o.get('pixel_id'):pixels.add(o['pixel_id'])
  for k in ['id','task_id','internal_sample_id']:
   if isinstance(o.get(k),str) and o[k] in roles:
    x=roles[o[k]];groups.add(x['group_id']);pixels.add(x['pixel_id'])
  for v in o.values():collect(v)
 elif isinstance(o,list):
  for x in o:collect(x)
def read(p):
 p=Path(p);collect([json.loads(x) for x in p.read_text().splitlines()] if p.suffix=='.jsonl' else json.loads(p.read_text()));sources.append(str(p))
prev=json.loads((Path(P['previous_graph_run'])/'private_membership.json').read_text())
for s in prev['exclusion_sources']:read(s)
for run in [P['previous_run'],P['previous_graph_run']]:
 read(Path(run)/'private_membership.json');read(Path(run)/'private_learning.json')
read(Path(P['extra_exclusion_run'])/'private_membership.json');read(Path(P['extra_exclusion_run'])/'private_learning.json')
read(Path(P['previous_quick_run'])/'private_membership.json');read(Path(P['previous_quick_run'])/'private_learning.json')
read(Path(P['previous_visual_run'])/'private_membership.json');read(Path(P['previous_visual_run'])/'private_learning.json')
parent=Path(P['parent_run']);read(parent/'private_membership.json');read(parent/'private_learning.json')
read(Path(P['previous_action_run'])/'private_membership.json');read(Path(P['previous_action_run'])/'private_learning.json')
read(Path(P['previous_evidence_run'])/'private_membership.json');read(Path(P['previous_evidence_run'])/'private_learning.json')
collect(json.loads(Path(P['old_state']).read_text())['memory'])
for task in ['visual_qa','classification','polyp_localization']:
 for x in sorted(map(json.loads,(D/f'visible/{task}/validation.jsonl').read_text().splitlines()),key=lambda x:x['id'])[:4]:collect(x)
types=['yes_no_or_presence','count','color_set','instrument','location_set','choice_or_other'];meta={x['id']:x['metadata'] for x in map(json.loads,(D/'private/visual_qa/h_search.jsonl').read_text().splitlines())};visible=list(map(json.loads,(D/'visible/visual_qa/h_search.jsonl').read_text().splitlines()));pools={};quotas={}
for role in ['selection']:
 for t in types:
  k=role+':'+t;pools[k]=collections.defaultdict(list);quotas[k]=(30 if t==types[0] else 14) if role=='confirmation' else 8 if role in ('train','selection') else 4
 for x in visible:
  m=meta[x['id']]
  if m['group_id'] not in groups and m['pixel_id'] not in pixels:pools[role+':'+m['question_type']][m['group_id']].append({**x,'metadata':m})
chosen=allocate(pools,quotas,P['seed']);members={k:[{'id':x['id'],'group_id':x['metadata']['group_id'],'pixel_id':x['metadata']['pixel_id']} for x in v] for k,v in chosen.items()};flat=[x for v in members.values() for x in v]
assert len(flat)==48==len({x['group_id'] for x in flat})==len({x['pixel_id'] for x in flat})
assert not {x['group_id'] for x in flat}&groups and not {x['pixel_id'] for x in flat}&pixels
(R/'private_membership.json').write_text(json.dumps({'seed':P['seed'],'members':members,'excluded_groups':len(groups),'exclusion_sources':sources},indent=2))
labels={x['id']:x['answer'] for x in map(json.loads,(D/'private/visual_qa/h_search.jsonl').read_text().splitlines())};data={}
for role in ['selection']:
 data[role]=[]
 for i in range(max(quotas[role+':'+t] for t in types)):
  for t in types:
   rows=chosen[role+':'+t]
   if i>=len(rows):continue
   x=rows[i];m=dict(x['metadata']);m.update(original_role=m['role'],study_role=role,role='h_search' if role=='train' else 'validation');answer=labels[x['id']];assert isinstance(answer,str) and answer.strip()
   data[role].append({**x,'metadata':m,'answer':answer,'task_id':x['id'],'task_name':'visual_qa','task_type':'medical_visual_qa','split':role})
data['train']=json.loads((parent/'private_learning.json').read_text())['train']
fixed=[next(x for x in data['train'] if x['metadata']['question_type']==t) for t in types]
prior=Path(P['previous_evidence_run']);assert json.loads((prior/'SUPERVISOR_EXIT.json').read_text())['success'] and not json.loads((prior/'EVIDENCE_GATE.json').read_text())['accepted']
assert not any(prior.glob('confirm_*')),'previous confirmation may have been evaluated'
confirmation=json.loads((prior/'private_confirmation.json').read_text());assert len(confirmation)==100 and len({x['task_id'] for x in confirmation})==100
oldmembers=json.loads((prior/'private_membership.json').read_text())['members'];held=[x for k,v in oldmembers.items() if k.startswith('confirmation:') for x in v];assert len(held)==100 and {x['id'] for x in held}=={x['task_id'] for x in confirmation}
assert not {x['group_id'] for x in held}&{x['group_id'] for x in flat} and not {x['pixel_id'] for x in held}&{x['pixel_id'] for x in flat}
(R/'private_confirmation.json').write_text(json.dumps(confirmation))
member=json.loads((R/'private_membership.json').read_text());member['members'].update({k:v for k,v in oldmembers.items() if k.startswith('confirmation:')});member['confirmation_transfer_source']=str(prior);(R/'private_membership.json').write_text(json.dumps(member,indent=2))
(R/'CONFIRMATION_TRANSFER.json').write_text(json.dumps({'count':100,'prior_gate_rejected':True,'prior_confirm_workers_absent':True,'confirmation_only':True,'unchanged_reserved_membership':True,'disjoint_from_new_selection':True,'no_model_outputs_used_for_transfer':True},indent=2))
(R/'private_learning.json').write_text(json.dumps({**data,'fixed_examples':[{'question':x['question'],'answer':x['answer']} for x in fixed]}))
receipt={'counts':{'reused_parent_train':48,'new_selection':48,'transferred_unevaluated_confirmation':100},'new_selection_disjoint_from_all_prior_used_and_reserved_groups':True,'transferred_confirmation_never_evaluated':True,'excluded_groups':len(groups),'original_test_untouched':True,'patient_independence':'unverified','model_calls':0}
(R/'DATA_CHECK.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
