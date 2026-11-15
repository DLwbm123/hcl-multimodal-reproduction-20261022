"""Training-only paired mechanism audit; never select or deploy a winner."""
import json
from pathlib import Path
from contracts import rows,score_map
from adaptive import intent

def analyze(root):
 r=Path(root);p=json.loads((r/'protocol.json').read_text());data=json.loads((r/'private_learning.json').read_text());assert set(data)=={'train'}
 train={str(e['task_id']):e for e in data['train']};spatial={k for k,e in train.items() if intent(e['question']) in {'color','location'}};assert len(train)==48 and len(spatial)==16
 assert not (r/'private_confirmation.json').exists() and not any(r.glob('eval_*')) and not any(r.glob('confirm_*'))
 out={'diagnostic_only':True,'policy_updated':False,'selection_n':0,'confirmation_n':0,'roles':{},'paired':{}};maps={};props={};traces={}
 for role in p['roles']:
  d=r/role;assert json.loads((d/'EXIT.json').read_text())['exit_code']==0;macro=role in {'fit_crop','fit_fullview'};expected=spatial if macro else set(train)
  m,metrics=score_map(d,'training_diagnostic');assert set(m)==expected;maps[role]=m
  ro=rows(d/'read_only_checks.jsonl');assert len(ro)==1 and ro[0]['count']==len(expected) and ro[0]['unchanged'] and ro[0]['weights_frozen']
  audit=[x for x in rows(d/'private_retrieval_audit.jsonl') if x['scope'].startswith('training_diagnostic:')];assert {x['query_id'] for x in audit}==expected
  for x in audit:
   assert set(x['source_ids'])<=set(train) and x['query_id'] not in x['source_ids'];assert x['grounded_macro']==macro and x['fullview_control']==(role=='fit_fullview')
  ep=[x for x in rows(d/'private_graph_traces.jsonl') if x['scope'].startswith('training_diagnostic:')];assert len(ep)==len(expected);traces[role]={x['sample_id']:x for x in ep};props[role]={};geometry={'ok':0,'absent':0,'uncertain':0,'error':0};successful=0;fallback=0
  for x in ep:
   events=[e for e in x['events'] if e['node']=='execute'];assert len(events)<=2
   if macro:
    assert x['controller_calls']==0 and len(events)==2 and events[0]['tool']=='propose_target_region'
    prop=json.loads(events[0]['result']);props[role][x['sample_id']]=prop;geometry[prop['status']]+=1
    if prop['status']=='ok':
     box=[0,0,*prop['original_size']] if role=='fit_fullview' else prop['crop_bbox_original'];assert events[1]['tool']=='inspect_region' and events[1]['args']['bbox']==box
     obs=json.loads(events[1]['result']);assert obs['status']=='ok' and obs['crop_bbox_original']==box;successful+=1
    else:assert events[1]['tool']=='answer_visual_question';fallback+=1
  calls=rows(d/'call_costs.jsonl');assert len(calls)==json.loads((d/'model_attempts.json').read_text())['attempts'];features=rows(d/'private_feature_costs.jsonl');all_ep=rows(d/'private_graph_traces.jsonl')
  out['roles'][role]={'metrics':metrics,'training_n':len(m),'training_correct':sum(m.values()),'training_graph_failed':sum(x['status']!='completed' for x in ep),'geometry':geometry,'actual_inspections':successful,'fallbacks':fallback,'generation_calls':len(calls),'generation_failed':sum(not x['success'] for x in calls),'generation_seconds':sum(x['seconds'] for x in calls),'feature_forwards':len(features),'feature_seconds':sum(x['seconds'] for x in features),'tool_calls':sum(e['node']=='execute' for x in all_ep for e in x['events']),'controller_calls':sum(e['node']=='process' for x in all_ep for e in x['events']),'read_only_and_weights_frozen':True}
 for name,left,right,ids in [('unchanged_repeat','fit_repeat_a','fit_repeat_b',set(train)),('view_control','fit_crop','fit_fullview',spatial),('crop_vs_incumbent','fit_repeat_a','fit_crop',spatial),('fullview_vs_incumbent','fit_repeat_a','fit_fullview',spatial)]:
  a=maps[left];b=maps[right];v={'n':len(ids),'left_correct':sum(a[k] for k in ids),'right_correct':sum(b[k] for k in ids),'gains':sum(b[k] and not a[k] for k in ids),'losses':sum(a[k] and not b[k] for k in ids),'answer_changed':sum(traces[left][k]['answer']!=traces[right][k]['answer'] for k in ids),'by_intent':{}}
  for k in ids:
   q=intent(train[k]['question']);g=v['by_intent'].setdefault(q,{'n':0,'gains':0,'losses':0});g['n']+=1;g['gains']+=int(b[k] and not a[k]);g['losses']+=int(a[k] and not b[k])
  if name=='view_control':
   matched={k for k in ids if props[left][k]==props[right][k]};v['identical_proposal_n']=len(matched);v['identical_proposal_gains']=sum(b[k] and not a[k] for k in matched);v['identical_proposal_losses']=sum(a[k] and not b[k] for k in matched);v['unmatched_n']=len(ids-matched)
  out['paired'][name]=v
 out['cost_totals']={k:sum(v[k] for v in out['roles'].values()) for k in ['generation_calls','generation_failed','generation_seconds','feature_forwards','feature_seconds','tool_calls','controller_calls']};out['limitations']=p['limitations'];(r/'ANALYSIS.json').write_text(json.dumps(out,indent=2)+'\n');return out
