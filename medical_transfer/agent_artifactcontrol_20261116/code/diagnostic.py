"""Training-only artifact transport invariance and paired outcome audit."""
import json
from pathlib import Path
from contracts import rows,score_map

def analyze(root):
 r=Path(root);p=json.loads((r/'protocol.json').read_text());data=json.loads((r/'private_learning.json').read_text());assert set(data)=={'train'};ids={str(e['task_id']) for e in data['train']};assert len(ids)==48 and not (r/'private_confirmation.json').exists() and not any(r.glob('eval_*')) and not any(r.glob('confirm_*'))
 out={'diagnostic_only':True,'selection_n':0,'confirmation_n':0,'policy_fitted':False,'roles':{},'paired':{}};maps={};traces={}
 for role in p['roles']:
  d=r/role;assert json.loads((d/'EXIT.json').read_text())['exit_code']==0;clean=role!='fit_raw';m,metrics=score_map(d,'training_diagnostic');assert set(m)==ids;maps[role]=m
  ro=rows(d/'read_only_checks.jsonl');assert len(ro)==1 and ro[0]['count']==48 and ro[0]['unchanged'] and ro[0]['weights_frozen'];audit=[x for x in rows(d/'private_retrieval_audit.jsonl') if x['scope'].startswith('training_diagnostic:')];assert {x['query_id'] for x in audit}==ids
  for x in audit:assert set(x['source_ids'])<=ids and x['query_id'] not in x['source_ids'] and not x['grounded_macro'] and x['strip_artifact_names']==clean
  all_ep=rows(d/'private_graph_traces.jsonl');ep=[x for x in all_ep if x['scope'].startswith('training_diagnostic:')];assert len(ep)==48;traces[role]={x['sample_id']:x for x in ep};assert all(sum(e['node']=='execute' for e in x['events'])<=2 for x in ep)
  calls=rows(d/'call_costs.jsonl');assert len(calls)==json.loads((d/'model_attempts.json').read_text())['attempts'];features=rows(d/'private_feature_costs.jsonl');model_nodes=[x for x in calls if x['phase'] in ['graph_controller','graph_finalize']];exposure=sum(x['artifact_identifiers_in_prompt'] for x in model_nodes)
  if clean:assert exposure==0
  private_artifacts=sum('artifact' in json.loads(e['result']) for x in all_ep for e in x['events'] if e['node']=='execute' and e['tool'] in {'inspect_region','segmentation'})
  out['roles'][role]={'metrics':metrics,'training_n':48,'training_correct':sum(m.values()),'training_graph_failed':sum(x['status']!='completed' for x in ep),'model_node_artifact_identifier_occurrences':exposure,'model_node_calls_with_artifact_identifier':sum(x['artifact_identifiers_in_prompt']>0 for x in model_nodes),'raw_private_receipts_with_artifact':private_artifacts,'generation_calls':len(calls),'generation_failed':sum(not x['success'] for x in calls),'generation_seconds':sum(x['seconds'] for x in calls),'feature_forwards':len(features),'feature_seconds':sum(x['seconds'] for x in features),'tool_calls':sum(e['node']=='execute' for x in all_ep for e in x['events']),'controller_calls':sum(e['node']=='process' for x in all_ep for e in x['events']),'read_only_and_weights_frozen':True}
 for name,left,right in [('raw_to_clean','fit_raw','fit_clean_a'),('clean_repeat','fit_clean_a','fit_clean_b')]:
  a,b=maps[left],maps[right];out['paired'][name]={'n':48,'left_correct':sum(a.values()),'right_correct':sum(b.values()),'gains':sum(b[k] and not a[k] for k in ids),'losses':sum(a[k] and not b[k] for k in ids),'answer_changed':sum(traces[left][k]['answer']!=traces[right][k]['answer'] for k in ids)}
 out['cost_totals']={k:sum(v[k] for v in out['roles'].values()) for k in ['generation_calls','generation_failed','generation_seconds','feature_forwards','feature_seconds','tool_calls','controller_calls']};out['limitations']=p['limitations'];(r/'ANALYSIS.json').write_text(json.dumps(out,indent=2)+'\n');return out
