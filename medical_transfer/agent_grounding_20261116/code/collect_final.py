"""Final skill/geometry checks; expose aggregate outcomes, never private boxes/crops."""
import json
from pathlib import Path
from grounding import use_macro,ELIGIBLE
from adaptive import intent

def load(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
def collect(root):
 root=Path(root);a=load(root/'ANALYSIS.json');done=load(root/'SUPERVISOR_EXIT.json');p=load(root/'protocol.json');policy=a['policy'];g=a['gate'];assert done['success'] and all(v==0 for v in done['returncodes'].values())
 expected={'fit_incumbent':{'counterfactual_train':48},'fit_macro':{'counterfactual_train':48}}
 if policy['allowed_macro_intents']:expected.update({'eval_'+k:{'selection':24,'historical':72} for k in ['incumbent','macro','proposed']})
 else:assert g['reasons']==['no_supported_grounding_skills'] and not any(root.glob('eval_*'))
 if g['accepted']:expected.update({'confirm_'+k:{'confirmation':76} for k in ['incumbent','proposed','macro']});assert load(root/'FROZEN_SELECTION.json')['locked_before_confirmation']
 else:assert not any(root.glob('confirm_*'))
 assert a['confirmation_opened']==g['accepted'] and set(expected)==set(done['returncodes'])==set(a['roles'])
 if policy['allowed_macro_intents']:assert g['accepted']==(g['current_correct_gain']>=1 and g['historical_losses']<=2 and g['selection_format']>=.95)
 data=load(root/'private_learning.json');examples={'counterfactual_train':data['train'],'selection':data['selection']}
 if g['accepted']:examples['confirmation']=load(root/'private_confirmation.json')
 qmap={str(e['task_id']):e['question'] for es in examples.values() for e in es};ids={k:{str(e['task_id']) for e in es} for k,es in examples.items()};train_ids=ids['counterfactual_train'];assert len(train_ids)==48
 out={**a,'audits':{}};maps={};paths={}
 for role,splits in expected.items():
  d=root/role;arm=role.split('_',1)[1];assert load(d/'EXIT.json')['exit_code']==0;calls=rows(d/'call_costs.jsonl');assert len(calls)==load(d/'model_attempts.json')['attempts'];ro=rows(d/'read_only_checks.jsonl');assert {x['split']:x['count'] for x in ro}==splits and all(x['unchanged'] and x['weights_frozen'] for x in ro)
  scores=rows(d/'private_scores.jsonl');assert len(scores)==len(splits);traces=rows(d/'private_graph_traces.jsonl');audits=rows(d/'private_retrieval_audit.jsonl');check_count=0
  for x in audits:
   assert set(x['source_ids'])<=train_ids and x['query_id'] not in x['source_ids'] and len(x['source_ids'])==x['reference_count']<=2
   if x['query_id'] in qmap:assert x['grounded_macro']==use_macro(qmap[x['query_id']],arm,policy);check_count+=1
  rec={'generation_calls':len(calls),'generation_failed':sum(not x['success'] for x in calls),'known_query_macro_checks':check_count,'read_only_and_frozen':True,'training_only_references':True,'splits':{}};maps[role]={};paths[role]={}
  for score in scores:
   split=score['split'];m={str(x['task_id']):bool(x['success']) for x in score['case_results']};assert len(m)==len(score['case_results'])==splits[split]
   if split in ids:assert set(m)==ids[split]
   maps[role][split]=m;eps=[x for x in traces if x['scope'].split(':')[0]==split];ars=[x for x in audits if x['scope'].split(':')[0]==split];assert len(eps)==len(ars)==(24 if split=='historical' else splits[split]);macro_ids={x['query_id'] for x in ars if x['grounded_macro']};geom={k:0 for k in ['ok','absent','uncertain','error']};crop_ok=crop_errors=fallback=macro_count=0;paths[role][split]={}
   for ep in eps:
    sid=str(ep['sample_id']);steps=[ev for ev in ep['events'] if ev['node']=='macro_select'];assert bool(steps)==(sid in macro_ids)
    if not steps:continue
    tool=[ev for ev in ep['events'] if ev['node']=='execute'];assert len(steps)==len(tool)==2 and ep['controller_calls']==0 and tool[0]['tool']=='propose_target_region';macro_count+=1
    proposal=json.loads(tool[0]['result']);status=proposal['status'];geom[status]+=1;paths[role][split][sid]=status
    if status=='ok':
     assert tool[1]['tool']=='inspect_region';result=json.loads(tool[1]['result']);assert tool[1]['args']['bbox']==proposal['crop_bbox_original']
     if result['status']=='ok':assert result['crop_bbox_original']==proposal['crop_bbox_original'];crop_ok+=1
     else:assert result['status']=='error';crop_errors+=1
    else:assert tool[1]['tool']=='answer_visual_question';fallback+=1
   rec['splits'][split]={'total':len(m),'correct':sum(m.values()),'graph_episodes':len(eps),'completed':sum(x['status']=='completed' for x in eps),'failed':sum(x['status']!='completed' for x in eps),'tool_calls':sum(ev['node']=='execute' for x in eps for ev in x['events']),'macro_episodes':macro_count,'proposal_status':geom,'successful_crops':crop_ok,'crop_errors':crop_errors,'fallback_episodes':fallback}
  out['audits'][role]=rec
 actual={}
 for e in data['train']:
  sid=str(e['task_id']);k=intent(e['question']);x=maps['fit_incumbent']['counterfactual_train'][sid];y=maps['fit_macro']['counterfactual_train'][sid];v=actual.setdefault(k,{'support':0,'incumbent_correct':0,'macro_correct':0,'gains':0,'losses':0});v['support']+=1;v['incumbent_correct']+=x;v['macro_correct']+=y;v['gains']+=int(y and not x);v['losses']+=int(x and not y)
 assert actual==policy['training_groups'];rule=p['routing'];assert sorted(k for k,v in actual.items() if k in ELIGIBLE and v['support']>=rule['min_support'] and v['gains']-v['losses']>=rule['min_net_training_gain'] and v['losses']<=rule['max_training_losses'])==policy['allowed_macro_intents']
 out['gains_losses_by_macro_path']={}
 for role,split,base in [('fit_macro','counterfactual_train','fit_incumbent'),('eval_proposed','selection','eval_incumbent'),('confirm_proposed','confirmation','confirm_incumbent')]:
  if role not in maps:continue
  x=maps[base][split];y=maps[role][split];agg={}
  for sid,status in paths[role][split].items():
   v=agg.setdefault(status,{'n':0,'gains':0,'losses':0});v['n']+=1;v['gains']+=int(y[sid] and not x[sid]);v['losses']+=int(x[sid] and not y[sid])
  out['gains_losses_by_macro_path'][role]=agg
 out['completion']={k:done[k] for k in ['ended','elapsed_seconds','bytes','success']};out['total_costs']={k:sum(v[k] for v in a['roles'].values()) for k in ['generation_calls','generation_failed','generation_seconds','feature_forwards','feature_seconds','graph_completed','graph_failed','tool_calls','macro_queries']};out['data_isolation']=load(root/'DATA_CHECK.json');out['limitations']=p['limitations']+['Legal coordinates are not validated localization accuracy; crop/fallback subgroup effects are descriptive post-run diagnostics.']
 (root/'PUBLIC_AGGREGATES.json').write_text(json.dumps(out,indent=2)+'\n');return out
