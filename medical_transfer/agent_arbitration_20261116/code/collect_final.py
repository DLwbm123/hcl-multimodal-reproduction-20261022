"""Verify a completed empty-override result without opening reserved evaluation data."""
import json
from pathlib import Path
from arbitration import strategy
from adaptive import intent

def load(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines()]
def collect(root):
 root=Path(root);a=load(root/'ANALYSIS.json');done=load(root/'SUPERVISOR_EXIT.json');assert done['success'] and all(v==0 for v in done['returncodes'].values());assert not a['policy']['contract_override_intents'] and not a['confirmation_opened'];assert a['gate']['reasons']==['no_supported_contract_overrides'];assert not any(root.glob('eval_*')) and not any(root.glob('confirm_*'))
 train=load(root/'private_learning.json')['train'];q={str(x['task_id']):x['question'] for x in train};assert len(q)==48;inc=load(root/'INCUMBENT_POLICY.json');out={**a,'audits':{}};maps={}
 for arm in ['incumbent','contract']:
  role='fit_'+arm;d=root/role;assert load(d/'EXIT.json')['exit_code']==0;calls=rows(d/'call_costs.jsonl');assert len(calls)==load(d/'model_attempts.json')['attempts'];score=rows(d/'private_scores.jsonl');assert len(score)==1 and score[0]['split']=='counterfactual_train';m={str(x['task_id']):bool(x['success']) for x in score[0]['case_results']};assert set(m)==set(q) and len(score[0]['case_results'])==48;maps[arm]=m
  ro=rows(d/'read_only_checks.jsonl');assert len(ro)==1 and ro[0]['unchanged'] and ro[0]['weights_frozen'] and ro[0]['count']==48
  audit=rows(d/'private_retrieval_audit.jsonl')
  for x in audit:
   assert x['source_strategy']==strategy(q[x['query_id']],arm,inc,{}) and set(x['source_ids'])<=set(q) and x['query_id'] not in x['source_ids']
  tr=[x for x in rows(d/'private_graph_traces.jsonl') if x['scope'].startswith('counterfactual_train:')];assert len(tr)==48
  out['audits'][role]={'total':48,'correct':sum(m.values()),'generation_calls':len(calls),'generation_failed':sum(not x['success'] for x in calls),'source_choices_checked':len(audit),'completed':sum(x['status']=='completed' for x in tr),'failed':sum(x['status']!='completed' for x in tr),'training_only_references':True,'read_only_and_frozen':True}
 actual={}
 for sid,question in q.items():
  v=actual.setdefault(intent(question),{'support':0,'incumbent_correct':0,'contract_correct':0,'gains':0,'losses':0});a0=maps['incumbent'][sid];b=maps['contract'][sid];v['support']+=1;v['incumbent_correct']+=a0;v['contract_correct']+=b;v['gains']+=int(b and not a0);v['losses']+=int(a0 and not b)
 assert actual==a['policy']['training_groups']
 out['completion']={k:done[k] for k in ['ended','elapsed_seconds','bytes','success']};out['total_costs']={k:sum(v[k] for v in a['roles'].values()) for k in ['generation_calls','generation_failed','generation_seconds','feature_forwards','feature_seconds','graph_completed','graph_failed','tool_calls']};out['data_plan']=load(root/'DATA_CHECK.json');out['limitations']=load(root/'protocol.json')['limitations'];out['no_independent_attribution_claim']='Only training comparisons ran; zero supported overrides does not prove independent visual observations caused the earlier confirmation gain.'
 (root/'PUBLIC_AGGREGATES.json').write_text(json.dumps(out,indent=2)+'\n');return out
