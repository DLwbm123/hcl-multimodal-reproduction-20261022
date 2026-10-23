"""Frozen D1/D2 learning-only diagnostics. No HCL entry or candidate generation."""
import asyncio,json,time
from pathlib import Path
from adapter import load_upstream
from experiment import episode
from learning import atomic
from qwen_transport import LocalModel
from strict_scoring import value_match

def main(root,repo,hcl_path,model_path):
 import sys
 sys.path[:0]=[repo,hcl_path]
 root=Path(root);p=json.loads((root/'protocol.json').read_text())
 assert p['hcl_enabled'] is False and p['model_layers']==['D1','D2']
 d0=json.loads((root/'D0_SUMMARY.json').read_text());assert d0['D1_ready']
 selected=json.loads((root/'D0_SELECTION.private.json').read_text())['items']
 visible={c['id']:c for c in json.loads((root/'data/learning_visible.json').read_text())}
 from hcl.models import LocalTransformersModel
 backend=LocalTransformersModel(name='Qwen2.5-VL-7B-Instruct',path=model_path,torch_dtype='bfloat16',attn_implementation='sdpa',max_new_tokens=4096,answer_max_new_tokens=4096,temperature=0.,enable_thinking=False,local_files_only=True)
 model=LocalModel(backend,root);upstream=load_upstream(repo);layers={}
 for layer in ['D1','D2']:
  records=[]
  for i,ref in enumerate(selected):
   c=visible[ref['case_id']];a=ref['arguments']
   prompt='Learning-only diagnostic, NOT an autonomous benchmark or HCL trial. Complete only the single calculation specified below. Read the visible facts, execute the specified calculator, then explicitly submit its raw computed value by result_id. Do not solve other calculations in the records. Tool availability and the 24-turn limit are unchanged.\n'
   if layer=='D1':prompt+='The following exact arguments and evidence bundle are supplied as diagnostic assistance; no computed answer is supplied: '+json.dumps(a,ensure_ascii=False)
   else:prompt+='The exact calculator and scope are supplied, but you must extract all parameters from the visible records and cite fact_ids in input_sources: '+json.dumps({k:a[k] for k in ['calculator_id','patient_id','episode_id']})
   case={**c,'id':ref['id'],'prompt':prompt}
   atomic(root/'progress.json',{'phase':layer,'ordinal':i,'total':len(selected),'calls':model.calls,'time':time.time()})
   result=asyncio.run(episode(upstream,root/'episodes'/layer,case,model,[]))
   ledger=result['result_ledger'];matching=[r for r in ledger if r.get('calculator_id')==a['calculator_id'] and r.get('patient_id')==a['patient_id'] and r['episode_id']==a['episode_id']]
   target=[r for r in (result['answer'] or {}).get('calculators',[]) if r['name']==ref['formula_output']['name'] and r['episode_id']==a['episode_id']]
   tools=[t for t in result['trace'] if t['role']=='tool'];execs=[t for t in tools if t['name']=='execute_calculator']
   def correct_params(r):return r.get('parameters')==a['parameters']
   summary={'submitted':result['submitted'],'scoped_computations':len(matching),'exact_reference_parameters':any(correct_params(r) for r in matching),'reference_value_computed':any(value_match(r['value'],ref['raw_reference_value']) for r in matching),'reference_value_submitted':len(target)==1 and value_match(target[0]['value'],ref['raw_reference_value']),'execution_requests':len(execs),'execution_errors':sum(t['status']=='error' for t in execs),'tool_requests':len(tools),'model_calls':result['model_calls'],'termination':result['termination'],'cached_requests':result['duplicates_cached']}
   result['diagnostic_summary']=summary;atomic(root/'episodes'/layer/ref['id']/'RESULT.json',result);records.append(summary)
  aggregate={k:sum(r[k] for r in records) for k in ['submitted','scoped_computations','exact_reference_parameters','reference_value_computed','reference_value_submitted','execution_requests','execution_errors','tool_requests','model_calls','cached_requests']}
  aggregate.update(cases=len(records),terminations={t:sum(r['termination']==t for r in records) for t in sorted(set(r['termination'] for r in records))},computed_then_lost=sum(r['reference_value_computed'] and not r['reference_value_submitted'] for r in records))
  aggregate['advance_gate']=(aggregate['reference_value_submitted']>=p['D1_min_correct_submissions'] and aggregate['computed_then_lost']==0 and aggregate['exact_reference_parameters']>=p['D1_min_exact_parameters']) if layer=='D1' else None
  layers[layer]=aggregate;atomic(root/(layer+'_SUMMARY.json'),aggregate)
  if layer=='D1' and not aggregate['advance_gate']:
   atomic(root/'RESULT.json',{'complete':True,'layers':layers,'D2_skipped':'D1 gate failed','hcl_started':False,'model_calls':model.calls,'scope':'learning-only assisted diagnostic, not H0 acceptance'});return 2
 atomic(root/'RESULT.json',{'complete':True,'layers':layers,'D3_D4_skipped':'scope visibility not yet certified; no automatic progression','hcl_started':False,'model_calls':model.calls,'scope':'learning-only assisted diagnostic, not H0 acceptance'});return 0
