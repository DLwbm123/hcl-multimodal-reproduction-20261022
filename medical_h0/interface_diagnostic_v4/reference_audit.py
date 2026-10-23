"""Learning-only D0. No model, no test data, no benchmark-value-derived inputs."""
import json,sys,time
from pathlib import Path
from collections import Counter
from clinical_catalog import identity,coverage,CALCS,details
from reference_inputs import mapped_inputs,UnresolvedInput
from runtime_state import ExecutionState
from strict_scoring import value_match

def audit(root,repo):
 root=Path(root);sys.path.insert(0,str(repo))
 from medharness.scorers.calc import answer_episode_ids,derived_input_edges
 cases=json.loads((root/'data/learning_visible.json').read_text());gold=json.loads((root/'data/learning_gold.private.json').read_text());rows=[];references=[]
 for ci,c in enumerate(cases):
  task=gold[c['id']];eps=answer_episode_ids(task);edges=derived_input_edges(task)
  for ai,(a,ep) in enumerate(zip(task['calculator_answers'],eps)):
   cid=identity(a['name']);r={'case_ordinal':ci,'target_ordinal':ai,'calculator_id':cid,'name':a['name'],'output_episode_id':ep,'formula_enabled':cid in coverage(),'episode_exists_in_visible_facts':any(f['episode_id']==ep for f in c['rows']),'scope_source':'Evaluator dependency graph. Supplied only in oracle diagnostics; autonomous derivability not yet established.','dependency_count':sum(e['consumer_index']==ai for e in edges),'input_mapping':'not_reviewed','output_convention':'not_reviewed','reference_agrees_with_benchmark':None}
   if cid not in CALCS:r['status']='no_exact_registered_identity'
   elif cid not in coverage():r['status']='registered_not_enabled'
   else:
    try:
     params,used=mapped_inputs(cid,c['rows'],ep)
     evidence=[]
     for u in used:
      matches=[(i,f) for i,f in enumerate(c['rows']) if f['episode_id']==u['episode_id'] and f['field_name']==u['field'] and f['value_text']==u['value'] and f['unit']==u['unit']]
      if len(matches)!=1:raise UnresolvedInput('reference evidence is ambiguous')
      evidence.append((f'fact_{matches[0][0]+1:04d}',matches[0][1]))
     patients={f['patient_id'] for _,f in evidence}
     if len(patients)!=1:raise UnresolvedInput('reference evidence spans patients or is empty')
     sources={p:{'fact_ids':list(dict.fromkeys(i for i,_ in evidence)),'transformation':'Reference-only reviewed mapping from the attached evidence bundle; per-field derivation is in reference_inputs.py.'} for p in params}
     args={'calculator_id':cid,'episode_id':ep,'patient_id':next(iter(patients)),'input_sources':sources,'parameters':params}
     state=ExecutionState(c['rows']);state.call('read_facts',{})
     out,error=state.call('execute_calculator',args)
     r['input_mapping']='visible_fact_reference_map';r['unit_mapping']='mapping code and source units retained; no clinical certification';r['used_sources']=used
     if error:raise UnresolvedInput('reference execution: '+json.dumps(out))
     submission,err=state.call('submit_answer',{'calculators':[{'result_id':out['result_id']}]})
     if err:raise RuntimeError('reference serialization failed: '+str(submission))
     r['status']='reference_path_executed';r['output_convention']='raw_library_value_no_gold_conversion';r['raw_value']=out['value']
     references.append({'id':f'diagnostic_{ci}_{ai}','case_id':c['id'],'case_ordinal':ci,'target_ordinal':ai,'arguments':args,'raw_reference_value':out['value'],'formula_output':out,'original_prompt':c['prompt']})
    except (UnresolvedInput,ValueError,TypeError,KeyError) as e:r['status']='reference_mapping_or_contract_blocked';r['blocker']=str(e)
   rows.append(r)
 # Freeze a small, structurally selected set BEFORE comparing reference outputs with target values.
 selected=[];seen=set()
 for ref in references:
  if ref['case_ordinal'] not in seen:selected.append(ref);seen.add(ref['case_ordinal'])
 for ref in references:
  if len(selected)>=6:break
  if ref not in selected:selected.append(ref)
 selected=selected[:6]
 frozen={'created_epoch':time.time(),'selection':'first successful reference path per learning case in frozen order; fill to six in same order; no benchmark scores used','items':selected}
 path=root/'D0_SELECTION.private.json'
 if path.exists():raise RuntimeError('refuse to replace reference selection')
 path.write_text(json.dumps(frozen,indent=2))
 for row in rows:
  if row['status']=='reference_path_executed':
   target=gold[cases[row['case_ordinal']]['id']]['calculator_answers'][row['target_ordinal']]
   row['reference_agrees_with_benchmark']=value_match(row['raw_value'],target['final_answer'])
 selected_keys={(r['case_ordinal'],r['target_ordinal']) for r in selected}
 selected_rows=[r for r in rows if (r['case_ordinal'],r['target_ordinal']) in selected_keys]
 summary={'cases':len(cases),'targets':len(rows),'status_counts':dict(Counter(r['status'] for r in rows)),'formula_enabled_targets':sum(r['formula_enabled'] for r in rows),'reference_value_agreements':sum(r['reference_agrees_with_benchmark'] is True for r in rows),'selected_count':len(selected),'selected_value_agreements':sum(r['reference_agrees_with_benchmark'] is True for r in selected_rows),'D1_ready':len(selected)>=4 and all(r['reference_agrees_with_benchmark'] is True for r in selected_rows),'D3_D4_ready':False,'scope_visibility_status':'not established; do not run unassisted scope/planning diagnostics yet','no_model_calls':True}
 (root/'D0_MATRIX.private.json').write_text(json.dumps(rows,indent=2));(root/'D0_SUMMARY.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary));print(json.dumps([{'case':r['case_ordinal'],'target':r['target_ordinal'],'calculator':r['calculator_id'],'status':r['status'],'blocker':r.get('blocker'),'agree':r['reference_agrees_with_benchmark']} for r in rows if r['formula_enabled']],indent=2))
if __name__=='__main__':audit(*sys.argv[1:])
