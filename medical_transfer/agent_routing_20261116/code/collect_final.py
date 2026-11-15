"""Final checks of empirical routes, role denominators and private-data boundaries."""
import json
from pathlib import Path
from adaptive import intent

def load(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
def route_expected(arm,question,allowed):return arm=='evidence' or (arm=='routed' and intent(question) in allowed)
def collect(root):
 root=Path(root);done=load(root/'SUPERVISOR_EXIT.json');assert done['success'] and all(v==0 for v in done['returncodes'].values())
 a=load(root/'ANALYSIS.json');policy=a['policy'];g=a['gate'];p=load(root/'protocol.json');data=load(root/'private_learning.json');train_ids={str(e['task_id']) for e in data['train']};assert len(train_ids)==48
 expected={'fit_base':{'counterfactual_train':48},'fit_evidence':{'counterfactual_train':48}}
 if policy['allowed_intents']:expected.update({'eval_'+k:{'selection':48,'historical':72} for k in ['base','evidence','routed']})
 if g['accepted']:expected.update({'confirm_'+k:{'confirmation':100} for k in ['base','routed','evidence']})
 else:assert not any(root.glob('confirm_*'))
 assert set(expected)==set(done['returncodes'])==set(a['roles'])
 if not policy['allowed_intents']:assert g['reasons']==['no_supported_training_routes'] and not any(root.glob('eval_*'))
 else:assert g['accepted']==(g['current_correct_gain']>=1 and g['historical_losses']<=2 and g['selection_format']>=.95)
 assert a['confirmation_opened']==g['accepted']
 rule=p['routing'];allowed=sorted(k for k,v in policy['training_groups'].items() if v['support']>=rule['min_support'] and v['gains']-v['losses']>=rule['min_net_training_gain'] and v['losses']<=rule['max_training_losses']);assert allowed==policy['allowed_intents']
 examples={'counterfactual_train':data['train'],'selection':data['selection']}
 if g['accepted']:examples['confirmation']=load(root/'private_confirmation.json');assert load(root/'FROZEN_SELECTION.json')['locked_before_confirmation']
 qmap={str(e['task_id']):e['question'] for es in examples.values() for e in es};ids={k:{str(e['task_id']) for e in es} for k,es in examples.items()}
 out={**a,'audits':{}};maps={}
 for role,splits in expected.items():
  d=root/role;arm=role.split('_',1)[1];assert load(d/'EXIT.json')['exit_code']==0
  calls=rows(d/'call_costs.jsonl');issued=load(d/'model_attempts.json')['attempts'];assert issued==len(calls)
  ro=rows(d/'read_only_checks.jsonl');assert {x['split']:x['count'] for x in ro}==splits and all(x['unchanged'] and x['weights_frozen'] for x in ro)
  scores=rows(d/'private_scores.jsonl');assert len(scores)==len(splits);audits=rows(d/'private_retrieval_audit.jsonl');traces=rows(d/'private_graph_traces.jsonl');checked=0
  for x in audits:
   assert set(x['source_ids'])<=train_ids and x['query_id'] not in x['source_ids'];assert x['reference_count']==len(x['source_ids'])<=2
   if x['query_id'] in qmap:assert x['independent_selected']==route_expected(arm,qmap[x['query_id']],allowed);checked+=1
  rec={'generation_issued':issued,'generation_completed':len(calls),'generation_failed':sum(not x['success'] for x in calls),'known_query_route_checks':checked,'references_train_only_and_no_self':True,'splits':{}};maps[role]={}
  for x in scores:
   split=x['split'];m={str(v['task_id']):bool(v['success']) for v in x['case_results']};assert len(m)==len(x['case_results'])==splits[split]
   if split in ids:assert set(m)==ids[split]
   maps[role][split]=m;eps=[e for e in traces if e['scope'].split(':')[0]==split];ar=[e for e in audits if e['scope'].split(':')[0]==split];assert len(eps)==len(ar)==(24 if split=='historical' else splits[split])
   rec['splits'][split]={'total':len(m),'correct':sum(m.values()),'graph_episodes':len(eps),'completed':sum(e['status']=='completed' for e in eps),'failed':sum(e['status']!='completed' for e in eps),'independent_selected':sum(e['independent_selected'] for e in ar),'tool_calls':sum(ev['node']=='execute' for e in eps for ev in e['events'])}
  out['audits'][role]=rec
 base=maps['fit_base']['counterfactual_train'];evidence=maps['fit_evidence']['counterfactual_train'];actual={}
 for e in data['train']:
  k=intent(e['question']);sid=str(e['task_id']);v=actual.setdefault(k,{'support':0,'base_correct':0,'evidence_correct':0,'gains':0,'losses':0});v['support']+=1;v['base_correct']+=base[sid];v['evidence_correct']+=evidence[sid];v['gains']+=int(evidence[sid] and not base[sid]);v['losses']+=int(base[sid] and not evidence[sid])
 assert actual==policy['training_groups']
 out['completion']={k:done[k] for k in ['ended','elapsed_seconds','bytes','success']}
 out['data_isolation']=load(root/'DATA_CHECK.json');out['confirmation_transfer']=load(root/'CONFIRMATION_TRANSFER.json')
 out['total_costs']={k:sum(v[k] for v in a['roles'].values()) for k in ['generation_calls','generation_failed','generation_seconds','feature_forwards','feature_seconds','graph_completed','graph_failed','tool_calls','independent_selected']}
 out['limitations']=p['limitations']+['Whole role costs include preflight; concurrent GPU load confounds timing. Routing switches both tool guidance and final output contract; the trial does not isolate those two components.']
 (root/'PUBLIC_AGGREGATES.json').write_text(json.dumps(out,indent=2)+'\n');return out

if __name__=='__main__':
 assert not route_expected('base','What color is visible?',['color'])
 assert route_expected('routed','What color is visible?',['color'])
 assert not route_expected('routed','How many instruments?',['color'])
 assert route_expected('evidence','How many instruments?',[])
 print('Route-boundary self-check passed. Final audit requires all roles complete.')
