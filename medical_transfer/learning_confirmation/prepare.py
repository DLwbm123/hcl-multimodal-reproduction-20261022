import os
import json,random,time,sys,shutil
from pathlib import Path
C=Path(os.environ['CAMPAIGN_ROOT']);R=C/'runs/learning_confirmation_v1';P=Path(os.environ['PREPARED_DATA']);OLD=Path(os.environ['PILOT_PREPARED_DATA'])
S=C/'runs/semantic_boundary_v1'
assert not R.exists();R.mkdir();[(R/n).mkdir() for n in ['code','tmp','cache']]
rows=lambda p:[json.loads(x) for x in p.read_text().splitlines()]
def write(n,x):(R/n).write_text(json.dumps(x,indent=2))
tasks=['classification','polyp_localization','visual_qa'];roles={x['id']:x for x in rows(P/'audit/roles.jsonl')}
exclude={x['group_id'] for x in json.loads((S/'private_frozen_confirmation_membership.json').read_text())['cases']}
exclude.update(x['metadata']['group_id'] for x in rows(S/'private_screen.jsonl'))
visible={t:rows(P/f'visible/{t}/validation.jsonl') for t in tasks}
for t in tasks:
 exclude.update(roles[x['id']]['group_id'] for x in rows(OLD/f'visible/{t}/validation.jsonl'))
 exclude.update(roles[x['id']]['group_id'] for x in sorted(visible[t],key=lambda x:x['id'])[:4])
used=set(exclude);rng=random.Random(2026102302);selection=[]
for t in tasks:
 candidates=sorted(visible[t],key=lambda x:x['id']);rng.shuffle(candidates);chosen=[]
 for v in candidates:
  g=roles[v['id']]['group_id']
  if g in used:continue
  chosen.append({'task':t,'id':v['id'],'group_id':g});used.add(g)
  if len(chosen)==50:break
 assert len(chosen)==50;selection+=chosen
write('private_frozen_membership.json',{'seed':2026102302,'cases':selection,'frozen_before_answer_join':True,'excluded_groups':len(exclude)})
joined=[]
for t in tasks:
 ids={x['id'] for x in selection if x['task']==t};labels={x['id']:x for x in rows(P/f'private/{t}/validation.jsonl') if x['id'] in ids}
 for v in visible[t]:
  if v['id'] in ids:joined.append({**v,'task_id':v['id'],'task_name':t,'task_type':'medical_'+t,'split':'validation','answer':labels[v['id']]['answer'],'metadata':labels[v['id']]['metadata']})
(R/'private_confirmation.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in joined))
sys.path.insert(0,os.environ['MEDICAL_SOURCE'])
from medical_hcl.data import StrictMedicalLoader
loader=StrictMedicalLoader(OLD);learning={}
for t in tasks:
 learning[t]={s:loader.load_task_spec({'task_name':t},split=s,limit=24) for s in ['train','validation','anchor']}
 for s,es in learning[t].items():assert len(es)==24 and not ({x['metadata']['group_id'] for x in es}&{x['group_id'] for x in selection})
write('private_learning.json',learning);shutil.copy2(P/'class_vocabulary.json',R/'class_vocabulary.json')
write('protocol.json',{'name':'learning_confirmation_v1','tasks':tasks,'arms':['direct','static_no_memory','hcl_contract','hcl_semantic'],'hypothesis':'the fixed semantic boundary reduces interference during genuine gated HCL learning; matched HCL control isolates boundary contribution','model':'Qwen2.5-VL-7B BF16 temperature0 answer512 max2048 SDPA frozen','H_per_task':24,'validation_per_task':24,'anchors_per_task':24,'confirmation_per_task':50,'batchsize':24,'candidates_per_component':3,'components':['task_interface_structuring','router_workflow','memory_selector','router_context'],'B':0,'strict_improvement':True,'format_threshold':1.0,'template_contract':'same opt-in structural contract in both HCL arms; no extra retries','semantic_policy':'same unchanged final-context boundary as prior failed cross-task confirmation; no task-specific exception','learning_memory':'fresh empty full raw/abstract/skills; original capacities10/2, independent retention anchors24','coordinate_interface':'common dev_v2 in all arms; prior synthetic precision 0/4 failure retained','confirmation_role':'new validation content groups, frozen before labels; report only after stages, never gate/candidate/checkpoint selection','selection_data':'same original pilot H24/validation24/anchor24; no prior semantic confirmation data reused','final_gate':'HCL semantic VQA >= Static +5pp; no task accuracy/format below Static or HCL control; pooled paired95% interval lower>0 vs Static; report Direct and control and exact paired counts regardless; semantic contribution also compared to control','failure_policy':'only existing memory-selector schema exhaustion becomes whole-example failure; other exceptions fatal, no automatic restart','test_used':False,'patient_independence':'not established','max_new_model_calls':24600,'stages':3})
now=time.time();cb=json.loads((C/'campaign_budget.json').read_text());mb=json.loads((S/'budget.json').read_text())
write('budget.json',{'started_epoch':now,'early_stop_epoch':min(now+24600,cb['early_stop_epoch']),'hard_deadline_epoch':min(now+25200,cb['deadline_epoch']),'max_calls_per_hcl':12000,'max_calls_baselines':600,'early_stop_bytes':1800000000,'max_new_bytes':2000000000,'model_path':mb['model_path']})
print(json.dumps({'selected':len(selection),'learning_tasks':len(learning),'started_epoch':now}))
