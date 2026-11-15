"""Final allowlisted aggregates; validate provenance without exporting case data."""
import json
from pathlib import Path

def load(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
def check_retrieval(records,train_ids):
 for x in records:
  assert set(x['source_ids'])<=train_ids,'reference outside training registry'
  assert x['query_id'] not in x['source_ids'],'query self-reference'
  assert len(x['source_ids'])==x['reference_count']<=2

def collect(root):
 root=Path(root);done=load(root/'SUPERVISOR_EXIT.json');assert done['success'] and all(x==0 for x in done['returncodes'].values())
 a=load(root/'ANALYSIS.json');gate=a['gate'];accepted=gate['accepted'];assert accepted==a['confirmation_opened']
 assert accepted==(gate['current_correct_gain']>=1 and gate['historical_losses']<=2 and gate['selection_format']>=.95)
 train=load(root/'private_learning.json')['train'];train_ids={str(e['task_id']) for e in train};assert len(train_ids)==48
 expected={'eval_'+k:{'selection':48,'historical':72} for k in ['base','contract','evidence']}
 if accepted:
  expected.update({'confirm_'+k:{'confirmation':100} for k in ['base','contract','evidence','shifted']})
  assert load(root/'FROZEN_SELECTION.json')['locked_before_confirmation']
 else:assert not any(root.glob('confirm_*'))
 assert set(done['returncodes'])==set(expected)
 out={k:a[k] for k in ['gate','training_contract_support','training_vocabulary_counts','roles','limitations','confirmation_opened']}
 if accepted:out['paired_evidence_vs']=a['paired_evidence_vs']
 out['audits']={};retrieval={};confirmation_ids={str(e['task_id']) for e in load(root/'private_confirmation.json')} if accepted else None
 for role,splits in expected.items():
  d=root/role;assert load(d/'EXIT.json')['exit_code']==0
  calls=rows(d/'call_costs.jsonl');issued=load(d/'model_attempts.json')['attempts'];assert issued==len(calls) and all(x['success'] for x in calls)
  ro=rows(d/'read_only_checks.jsonl');assert len(ro)==len(splits) and all(x['unchanged'] and x['weights_frozen'] for x in ro)
  assert {x['split']:x['count'] for x in ro}==splits
  scores=rows(d/'private_scores.jsonl');assert len(scores)==len(splits);traces=rows(d/'private_graph_traces.jsonl');audits=rows(d/'private_retrieval_audit.jsonl');check_retrieval(audits,train_ids);retrieval[role]=audits
  rec={'generation_issued':issued,'generation_completed':len(calls),'read_only_and_weights_frozen':True,'training_only_references':True,'splits':{}}
  for x in scores:
   split=x['split'];case=x['case_results'];ids={str(e['task_id']) for e in case};assert len(ids)==len(case)==splits[split]
   if split=='confirmation':assert ids==confirmation_ids
   eps=[e for e in traces if e['scope'].split(':')[0]==split];ra=[e for e in audits if e['scope'].split(':')[0]==split]
   assert len(eps)==(24 if split=='historical' else splits[split])
   rec['splits'][split]={'total':len(case),'correct':sum(bool(e['success']) for e in case),'graph_episodes':len(eps),'completed':sum(e['status']=='completed' for e in eps),'failed':sum(e['status']!='completed' for e in eps),'tool_calls':sum(ev['node']=='execute' for e in eps for ev in e['events']),'queries_with_reference':sum(e['reference_count']>0 for e in ra),'queries_with_tool_statistics':sum(e['tool_evidence_present'] for e in ra)}
  out['audits'][role]=rec
 if accepted:
  shift=load(root/'confirm_shifted/SHIFT_CHECK.json');assert shift['current_images_changed']==100 and shift['cyclic_shift']==1
  def refs(role):return {x['query_id']:x['source_ids'] for x in retrieval[role] if x['scope'].split(':')[0]=='confirmation'}
  normal=refs('confirm_evidence');shifted=refs('confirm_shifted');assert set(normal)==set(shifted)==confirmation_ids
  out['shift_diagnostic']={'target_image_shift':1,'cases':100,'selected_reference_ids_changed':sum(normal[k]!=shifted[k] for k in normal),'reference_source_bank_unchanged':True,'interpretation':'Whole-Agent input perturbation. Current-image changes can also alter controller actions and adaptive reference ranking; not an isolated tool-vision or fixed-reference ablation. SHIFT_CHECK reference_images_unchanged refers to the source bank, not necessarily selected per-query references.'}
 out['total_costs']={k:sum(v[k] for v in a['roles'].values()) for k in ['generation_calls','generation_seconds','feature_forward_calls','feature_seconds','tool_calls','completed_episodes','failed_episodes']}
 out['completion']={k:done[k] for k in ['ended','elapsed_seconds','bytes','success']}
 out['data_isolation']=load(root/'DATA_CHECK.json')
 out['timing_limitation']='Shared GPU load; summed call seconds are not wall time or a controlled speed comparison. Full costs include preflight and failed episodes.'
 (root/'PUBLIC_AGGREGATES.json').write_text(json.dumps(out,indent=2)+'\n');return out

if __name__=='__main__':
 check_retrieval([{'source_ids':['a'],'query_id':'q','reference_count':1}],{'a'})
 for record in [{'source_ids':['test'],'query_id':'q','reference_count':1},{'source_ids':['a'],'query_id':'a','reference_count':1}]:
  try:check_retrieval([record],{'a'})
  except AssertionError:pass
  else:raise AssertionError('provenance check failed to reject leakage')
 print('Retrieval provenance checks passed; final collection requires all roles complete.')
