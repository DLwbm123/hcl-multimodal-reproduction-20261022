"""Allowlisted final aggregates with complete costs and matched-role checks."""
import json
from pathlib import Path

def load(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines()]
def collect(root):
 root=Path(root);a=load(root/'ANALYSIS.json');done=load(root/'SUPERVISOR_EXIT.json')
 assert done['success'] and all(v==0 for v in done['returncodes'].values())
 assert not a['gate']['accepted'] and not a['confirmation_opened']
 assert not any(root.glob('confirm_*'))
 out={k:a[k] for k in ['policy_candidate','gate','roles','limitations','confirmation_opened']};out['audited_roles']={}
 expected={'fit_full':{'counterfactual_train':48,'selection':24,'historical':72},'fit_one':{'counterfactual_train':48,'selection':24,'historical':72},'candidate':{'selection':24,'historical':72}}
 maps={}
 for role,expected_splits in expected.items():
  d=root/role;assert load(d/'EXIT.json')['exit_code']==0
  scores=rows(d/'private_scores.jsonl');assert len(scores)==len(expected_splits)
  maps[role]={};traces=rows(d/'private_graph_traces.jsonl');calls=rows(d/'call_costs.jsonl');issued=load(d/'model_attempts.json')['attempts']
  assert issued==len(calls) and all(x['success'] for x in calls)
  splits={}
  for score in scores:
   split=score['split'];m={v['task_id']:bool(v['success']) for v in score['case_results']};assert len(m)==expected_splits[split] and len(score['case_results'])==len(m);maps[role][split]=m
   eps=[e for e in traces if e['scope'].split(':')[0]==split]
   splits[split]={'total':len(m),'correct':sum(m.values()),'graph_episodes':len(eps),'graph_completed':sum(e['status']=='completed' for e in eps),'graph_failed':sum(e['status']!='completed' for e in eps),'tool_calls':sum(ev['node']=='execute' for e in eps for ev in e['events']),'learned_stops':sum(ev['node']=='learned_stop' for e in eps for ev in e['events'])}
  out['audited_roles'][role]={'generation_issued':issued,'generation_completed':len(calls),'splits':splits}
 out['paired_selection']={}
 for role in ['fit_one','candidate']:
  x=maps['fit_full']['selection'];y=maps[role]['selection'];assert set(x)==set(y)
  out['paired_selection'][role]={'gain':sum(y[k] and not x[k] for k in x),'loss':sum(x[k] and not y[k] for k in x)}
 out['completion']={k:done[k] for k in ['ended','elapsed_seconds','bytes','success']}
 out['total_costs']={k:sum(v[k] for v in a['roles'].values()) for k in ['generation_calls','generation_seconds','feature_forward_calls','feature_seconds','tool_calls','learned_stops']}
 check=load(root/'DATA_CHECK.json');out['data_isolation']={'reused_training':48,'new_selection':24,'reserved_confirmation':100,'official_test_accessed':False,'patient_independence':'unverified','check':check}
 out['cost_limitation']='Shared GPU load; elapsed times are not a controlled between-arm speed comparison. All preflight and failed-episode calls are included.'
 (root/'PUBLIC_AGGREGATES.json').write_text(json.dumps(out,indent=2)+'\n');return out
