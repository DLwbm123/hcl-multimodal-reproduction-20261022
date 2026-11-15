"""Partition an unevaluated reserve before any new inference; never use predictions."""
import os,json,random
from pathlib import Path
R=Path(os.environ['RUN_ROOT']);P=json.loads((R/'protocol.json').read_text());assert not (R/'private_membership.json').exists()
source=Path(P['previous_action_run']);assert json.loads((source/'SUPERVISOR_EXIT.json').read_text())['success'];assert not json.loads((source/'POLICY_GATE.json').read_text())['accepted'];assert not any(source.glob('confirm_*'))
examples=json.loads((source/'private_confirmation.json').read_text());member=json.loads((source/'private_membership.json').read_text());reserved={x['id']:x for k,v in member['members'].items() if k.startswith('confirmation:') for x in v};assert len(examples)==len(reserved)==100 and {e['task_id'] for e in examples}==set(reserved)
prior=[]
for run in [P['parent_run'],P['previous_evidence_run'],P['previous_routing_run']]:
 m=json.loads((Path(run)/'private_membership.json').read_text());prior.extend(x for v in m['members'].values() for x in v)
assert not {x['group_id'] for x in reserved.values()}&{x['group_id'] for x in prior}
assert not {x['pixel_id'] for x in reserved.values()}&{x['pixel_id'] for x in prior}
types=['yes_no_or_presence','count','color_set','instrument','location_set','choice_or_other'];rng=random.Random(P['seed']);selection=[];confirmation=[]
for kind in types:
 pool=sorted([e for e in examples if e['metadata']['question_type']==kind],key=lambda x:x['task_id']);rng.shuffle(pool);assert len(pool)>=4
 selection.extend(pool[:4]);confirmation.extend(pool[4:])
assert len(selection)==24 and len(confirmation)==76
for role,es in [('selection',selection),('confirmation',confirmation)]:
 for e in es:e['split']=role;e['metadata'].update(study_role=role,role='validation')
train=json.loads((Path(P['parent_run'])/'private_learning.json').read_text())['train'];assert len(train)==48 and not {e['task_id'] for e in train}&set(reserved)
(R/'private_learning.json').write_text(json.dumps({'train':train,'selection':selection,'fixed_examples':[]}));(R/'private_confirmation.json').write_text(json.dumps(confirmation))
m={'seed':P['seed'],'members':{role:[reserved[e['task_id']] for e in es] for role,es in [('selection',selection),('confirmation',confirmation)]},'transfer_source':str(source),'exclusion_sources':member['exclusion_sources']+[str(Path(P['previous_routing_run'])/'private_membership.json')]};flat=[x for v in m['members'].values() for x in v];assert len(flat)==len({x['group_id'] for x in flat})==len({x['pixel_id'] for x in flat})==100
(R/'private_membership.json').write_text(json.dumps(m,indent=2));receipt={'counts':{'reused_train':48,'selection_from_unopened_reserve':24,'confirmation_from_unopened_reserve':76},'prior_gate_rejected':True,'prior_confirmation_workers_absent':True,'prior100_preserved':True,'all100_distinct_groups_and_pixels':True,'disjoint_from_later_evidence_and_routing_panels':True,'selection_confirmation_disjoint':True,'model_calls':0,'patient_independence':'unverified','official_test_accessed':False};(R/'DATA_CHECK.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
