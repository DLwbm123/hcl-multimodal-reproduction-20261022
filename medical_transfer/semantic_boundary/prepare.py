import os
import json,random,time,shutil
from pathlib import Path
C=Path(os.environ['CAMPAIGN_ROOT']);R=C/'runs/semantic_boundary_v1'
D=Path(os.environ['DIAGNOSTIC_ROOT']);P=Path(os.environ['PREPARED_DATA'])
R.mkdir(exist_ok=False);(R/'code').mkdir();(R/'cache').mkdir();(R/'tmp').mkdir()
def rows(p):return [json.loads(x) for x in p.read_text().splitlines()]
def write(n,d):(R/n).write_text(json.dumps(d,indent=2))
tasks=['classification','polyp_localization','visual_qa'];roles={x['id']:x for x in rows(P/'audit/roles.jsonl')}
exclude={x['group_id'] for x in json.loads((D/'private_frozen_membership.json').read_text())['cases']}
visible={t:rows(P/f'visible/{t}/validation.jsonl') for t in tasks}
for t in tasks:
 old=Path(os.environ['PILOT_PREPARED_DATA'])/f'visible/{t}/validation.jsonl'
 exclude.update(roles[x['id']]['group_id'] for x in rows(old))
 exclude.update(roles[x['id']]['group_id'] for x in sorted(visible[t],key=lambda x:x['id'])[:4])
selection=[];used=set(exclude);rng=random.Random(20261023)
for t in tasks:
 candidates=sorted(visible[t],key=lambda x:x['id']);rng.shuffle(candidates);chosen=[]
 for v in candidates:
  g=roles[v['id']]['group_id']
  if g in used:continue
  chosen.append({'task':t,'id':v['id'],'group_id':g});used.add(g)
  if len(chosen)==50:break
 assert len(chosen)==50
 selection.extend(chosen)
write('private_frozen_confirmation_membership.json',{'seed':20261023,'cases':selection,'frozen_before_answer_join':True,'excluded_prior_used_validation_groups':len(exclude)})
# Only after membership is saved, join the selected validation labels. No test file is read.
joined=[]
for t in tasks:
 ids={x['id'] for x in selection if x['task']==t};labels={x['id']:x for x in rows(P/f'private/{t}/validation.jsonl') if x['id'] in ids}
 for v in visible[t]:
  if v['id'] in ids:joined.append({**v,'task_id':v['id'],'task_name':t,'task_type':'medical_'+t,'split':'validation','answer':labels[v['id']]['answer'],'metadata':labels[v['id']]['metadata']})
(R/'private_confirmation.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in joined))
for src,dest in [('private_development.jsonl','private_screen.jsonl'),('private_frozen_memory.json','private_frozen_memory.json'),('class_vocabulary.json','class_vocabulary.json')]:shutil.copy2(D/src,R/dest)
reference=[x for x in rows(D/'private_predictions.jsonl') if x['condition'] in ('static_no_memory','raw_only')]
(R/'private_screen_reference.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in reference))
write('protocol.json',{'name':'final_context_historical_memory_boundary_v1','tasks':tasks,'screen_per_task':12,'confirmation_per_task':50,'confirmation_seed':20261023,'conditions':['static_no_memory','raw_only','raw_semantic'],'single_change':'append fixed historical-memory evidence boundary to final context only when selected_memory nonempty','fixed_model':'Qwen2.5-VL-7B BF16 eval temperature0 answer512','memory_pool':'same prior accepted raw10; abstract/anchor/skills empty','retriever_unchanged':True,'candidate_search':False,'memory_learning':False,'screen_reuse':'existing static/raw outputs; only raw_semantic generates anew','confirmation_policy_frozen_before_outputs':True,'confirmation_runs_regardless_screen_score':True,'screen_success':'VQA raw_semantic >= raw accuracy; fewer losses of Static successes; no task accuracy or format loss vs raw','confirmation_success':'VQA raw_semantic >= raw and >= Static + 5pp; fewer losses of Static successes than raw; no task accuracy or format loss vs raw; additionally report paired bootstrap interval','test_used':False,'coordinate_semantic_gate':'prior 0/4 failed; localization exploratory','candidate_policies':1,'no_threshold_relaxation':True,'confirmation_not_for_further_tuning':True})
old=json.loads((D/'budget.json').read_text());now=time.time();end=min(now+6900,json.loads((C/'campaign_budget.json').read_text())['early_stop_epoch'])
write('budget.json',{'started_epoch':now,'early_stop_epoch':end,'hard_deadline_epoch':min(now+7200,json.loads((C/'campaign_budget.json').read_text())['deadline_epoch']),'max_calls':2500,'early_stop_bytes':450000000,'max_new_bytes':500000000,'model_path':old['model_path']})
print(json.dumps({'root':str(R),'screen':36,'confirmation':len(joined),'distinct_confirmation_groups':len({x['group_id'] for x in selection}),'budget_start':now}))
