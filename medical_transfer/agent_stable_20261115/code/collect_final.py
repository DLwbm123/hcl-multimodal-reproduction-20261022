"""Collect allowlisted public aggregates after the frozen study actually finishes."""
import json
from pathlib import Path

def load(p):return json.loads(p.read_text())
def rows(p):return [json.loads(s) for s in p.read_text().splitlines()]
def collect(root):
 root=Path(root);exit=load(root/'SUPERVISOR_EXIT.json');assert exit['success'] and all(v==0 for v in exit['returncodes'].values())
 p=load(root/'protocol.json');a=load(root/'ANALYSIS.json');assert len(a['confirmation'])==7 and all(m['total']==200 for m in a['confirmation'].values())
 out={k:a[k] for k in ['confirmation','paired','graph','costs','full_agent_learning_completed','limitations']};out['learning']={};out['costs_all_workers']={}
 offsets=load(root/'attempt_call_offsets.json');total_issued=total_completed=0
 for role in ['fresh','resume']+['confirm_'+x for x in p['confirmation']['arms']]:
  d=root/role;assert load(d/'EXIT.json')['exit_code']==0;issued=load(d/'model_attempts.json')['attempts'];c=rows(d/'call_costs.jsonl');archived=[];archive_issued=0
  for old in sorted(root.glob('attempt*/'+role)):
   archived+=rows(old/'call_costs.jsonl');archive_issued+=load(old/'model_attempts.json')['attempts']
  assert archive_issued==offsets.get(role,0), 'archived attempts must equal offsets already included in issued total'
  completed=len(c)+len(archived);assert completed<=issued
  out['costs_all_workers'][role]={'issued_calls_including_prior_attempts':issued,'completed_call_receipts':completed,'interrupted_or_missing_receipts':issued-completed,'successful_call_receipts':sum(x['success'] for x in c+archived),'model_seconds_including_archived_completed_calls':sum(x['seconds'] for x in c+archived)}
  total_issued+=issued;total_completed+=completed
  if role in ['fresh','resume']:
   original=a['learning'][role];records=rows(d/'learning_receipts.jsonl');gates=rows(d/f'state/{role}/evaluator/memory_update_history.jsonl');batches=[];seen=0
   for rec,gate in zip(records,gates):
    decisions=rec['decisions'][seen:];seen=len(rec['decisions'])
    batches.append({**{k:rec[k] for k in ['batch','before_version','after_version','memory_changed','memory_summary','candidate_count']},'training_episode_count':rec['graph_training']['episodes'],'training_completed':rec['graph_training']['completed'],'training_failed':len(rec['graph_training']['failures']),'component_decisions':[{'accepted':bool(x['accepted']),'reason':x['reason']} for x in decisions], 'memory_gate':{k:gate[k] for k in ['accepted','rejection_reasons','stable_current_task_metrics','candidate_current_task_metrics','candidate_historical_anchor_metrics']}})
   out['learning'][role]={k:v for k,v in original.items() if k!='batches'};out['learning'][role]['batches']=batches
 out['total_model_calls']={'issued':total_issued,'completed_receipts':total_completed,'interrupted_or_missing_receipts':total_issued-total_completed}
 out['completion']={k:exit[k] for k in ['ended','elapsed_seconds','bytes','success']}
 out['recovery_note']='A temporary-file size-accounting race interrupted the first confirmation attempt. All confirmation arms were rerun from the same frozen states; partial attempts and their calls were retained, and the original deadline was unchanged.'
 out['data_isolation']={'train':48,'selection':48,'development':80,'confirmation':200,'all_four_roles_new_source_and_pixel_groups':True,'patient_independence':'unverified','official_test_accessed':False}
 (root/'PUBLIC_AGGREGATES.json').write_text(json.dumps(out,indent=2)+'\n');return out
