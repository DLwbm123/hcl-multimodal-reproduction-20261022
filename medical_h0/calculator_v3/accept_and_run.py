"""Learning-only capability gate; no submission-only acceptance and no hidden retries."""
import asyncio,json,time
from pathlib import Path
from adapter import load_upstream
from experiment import episode,run
from learning import atomic
from qwen_transport import LocalModel
from strict_scoring import score_prediction
from clinical_catalog import identity,coverage

def summarize(results,roles):
    rows=[x for r in results for x in r['score']['rows']]
    selected=sum(x['selected'] for x in rows);correct=sum(x['ok'] for x in rows)
    s=[r['score']['selection'] for r in results]
    matched=sum(x['matched'] for x in s);pred=sum(x['predicted'] for x in s);gold=sum(x['gold'] for x in s)
    traces=[t for r in results for t in r['trace'] if t['role']=='tool']
    failures=sum(t.get('status')=='error' for t in traces)
    return {'cases':len(results),'submitted':sum(r['submitted'] for r in results),'outputs':len(rows),'correct':correct,
      'end_to_end_accuracy':correct/len(rows),'selection_precision':matched/pred if pred else 0.,'selection_recall':matched/gold if gold else 0.,
      'matched_outputs':selected,'conditional_value_accuracy':correct/selected if selected else 0.,
      'positive_cases_by_block':{role:sum(r['score']['n_correct']>0 for r in results if roles[r['case_id']]==role) for role in sorted(set(roles.values()))},
      'formula_covered_outputs':sum(r['score']['diagnostic']['formula_covered_outputs'] for r in results),
      'tool_errors':failures,'tool_calls':len(traces),'tool_error_rate':failures/len(traces) if traces else 1.,
      'execution_successes':sum(t.get('name')=='execute_calculator' and t.get('status')=='success' for t in traces),
      'submission_failures':sum(not r['submitted'] for r in results)}

def main(root,repo,hcl_path,model_path):
    import sys
    sys.path[:0]=[repo,hcl_path]
    from hcl.models import LocalTransformersModel
    backend=LocalTransformersModel(name='Qwen2.5-VL-7B-Instruct',path=model_path,torch_dtype='bfloat16',attn_implementation='sdpa',max_new_tokens=4096,answer_max_new_tokens=4096,temperature=0.,enable_thinking=False,local_files_only=True)
    model=LocalModel(backend,root);upstream=load_upstream(repo)
    visible=json.loads((root/'data/visible_split.json').read_text())
    chosen=[c for c in visible if c['role'].startswith('train_')];assert len(chosen)==8
    all_gold=json.loads((root/'data/evaluator_gold.json').read_text());gold={c['id']:all_gold[c['id']] for c in chosen};del all_gold
    roles={c['id']:c['role'] for c in chosen}
    def evaluate(cases,label):
        results=[]
        for i,c in enumerate(cases):
            atomic(root/'progress.json',{'phase':label,'ordinal':i,'calls':model.calls,'time':time.time()})
            r=asyncio.run(episode(upstream,root/label,c,model,[]))
            r['score']=score_prediction(r['answer'] or {'calculators':[]},gold[c['id']])
            atomic(root/label/c['id']/'RESULT.json',r);results.append(r)
        return results
    results=evaluate(chosen,'acceptance_unguided');summary=summarize(results,roles)
    gate=json.loads((root/'protocol.json').read_text())['acceptance_gate']
    passed=(summary['submitted']>=gate['min_submitted'] and summary['selection_precision']>=gate['min_selection_precision']
      and summary['selection_recall']>=gate['min_selection_recall'] and summary['end_to_end_accuracy']>=gate['min_end_to_end_accuracy']
      and summary['conditional_value_accuracy']>=gate['min_conditional_value_accuracy'] and summary['tool_error_rate']<=gate['max_tool_error_rate']
      and all(n>=gate['min_positive_cases_per_block'] for n in summary['positive_cases_by_block'].values()) and summary['execution_successes']>0)
    atomic(root/'ACCEPTANCE.json',{'passed':passed,'summary':summary,'thresholds':gate,'calls':model.calls,'time':time.time(),'forced_submission':False})
    if not passed:
        # Controlled diagnostic only: supply required identities on the SAME learning cases,
        # never numerical answers. This is not an acceptance retry or formal result.
        guided=[]
        for c in chosen:
            names=list(dict.fromkeys(a['name'] for a in gold[c['id']]['calculator_answers']))
            guided.append({**c,'prompt':c['prompt']+'\nLearning-only diagnostic: required calculator identities are supplied to isolate selection from input/execution/submission errors: '+json.dumps(names)+'. No numerical answers are supplied. Unsupported calculators must be reported honestly; do not substitute another version.'})
        controls=evaluate(guided,'diagnostic_oracle_names');other=summarize(controls,roles)
        atomic(root/'DIAGNOSTIC.json',{'unguided':summary,'oracle_names':other,'calls':model.calls,'formal_started':False,'interpretation':'Oracle names isolate a selection aid, not an oracle formula/input intervention; remaining errors need input/formula/submission trace review.'})
        return 2
    atomic(root/'FORMAL_STARTED.json',{'time':time.time(),'acceptance_calls':model.calls,'protocol':'calculator_v3','panel':'reused regression panel'})
    return run(root,repo,hcl_path,model_path,model_override=model)
