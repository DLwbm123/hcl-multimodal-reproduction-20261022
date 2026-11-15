"""Allowlisted final aggregates, including all learning and feature-extraction costs."""
import json
from pathlib import Path

def load(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines()]
def collect(root):
 root=Path(root);done=load(root/'SUPERVISOR_EXIT.json');assert done['success'] and all(c==0 for c in done['returncodes'].values());p=load(root/'protocol.json');a=load(root/'ANALYSIS.json')
 assert len(a['confirmation'])==6 and all(m['total']==100 for m in a['confirmation'].values())
 out={k:a[k] for k in ['confirmation','primary_GN_vs_GR','paired_GN_vs','graph','costs','limitations','full_agent_learning_completed']};out['learning']={};out['costs_all_workers']={};out['development']={}
 for role in ['fresh','reference']+['confirm_'+a for a in p['confirmation']['arms']]:
  d=root/role;assert load(d/'EXIT.json')['exit_code']==0;calls=rows(d/'call_costs.jsonl');issued=load(d/'model_attempts.json')['attempts'];assert len(calls)==issued and all(x['success'] for x in calls)
  f=d/'private_feature_costs.jsonl';features=rows(f) if f.exists() else []
  out['costs_all_workers'][role]={'generation_issued':issued,'generation_completed':len(calls),'generation_seconds':sum(x['seconds'] for x in calls),'feature_forwards':len(features),'feature_seconds':sum(x['seconds'] for x in features),'feature_counts':{k:sum(x['kind']==k for x in features) for k in ['text_forward','vision_forward']}}
  if role in ['fresh','reference']:
   original=a['learning'][role];recs=rows(d/'learning_receipts.jsonl');gates=rows(d/f'state/{role}/evaluator/memory_update_history.jsonl');assert len(recs)==len(gates)==2;batches=[];seen=0
   for rec,g in zip(recs,gates):
    decisions=rec['decisions'][seen:];seen=len(rec['decisions']);batches.append({**{k:rec[k] for k in ['batch','before_version','after_version','memory_changed','memory_summary','candidate_count']},'training_episode_count':rec['graph_training']['episodes'],'training_completed':rec['graph_training']['completed'],'training_failed':len(rec['graph_training']['failures']),'component_decisions':[{'accepted':bool(x['accepted']),'reason':x['reason']} for x in decisions],'memory_gate':{k:g[k] for k in ['accepted','rejection_reasons','stable_current_task_metrics','candidate_current_task_metrics','candidate_historical_anchor_metrics']}})
   out['learning'][role]={k:v for k,v in original.items() if k!='batches'};out['learning'][role]['batches']=batches
   out['development'][role]=next(x['metrics'] for x in rows(d/'private_scores.jsonl') if x['split']=='development')
 out['completion']={k:done[k] for k in ['ended','elapsed_seconds','bytes','success']};out['total_costs']={k:sum(x[k] for x in out['costs_all_workers'].values()) for k in ['generation_issued','generation_completed','generation_seconds','feature_forwards','feature_seconds']}
 out['data_isolation']={'train':48,'selection':48,'development':24,'confirmation':100,'all_roles_new_and_disjoint_source_and_pixel_groups':True,'prior_groups_excluded':load(root/'DATA_CHECK.json')['excluded_groups'],'patient_independence':'unverified','official_test_accessed':False}
 out['cost_limitation']='Wall times reflect concurrent GPU load; do not interpret between-arm timing as a controlled speed comparison.'
 (root/'PUBLIC_AGGREGATES.json').write_text(json.dumps(out,indent=2)+'\n');return out
