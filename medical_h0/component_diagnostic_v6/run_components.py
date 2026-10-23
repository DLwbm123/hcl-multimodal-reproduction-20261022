"""One ordered, gated synthetic component diagnostic. No HCL entry."""
import asyncio
import json
import sys
import time
from pathlib import Path
import bootstrap
from adapter_v5 import load_upstream
from adapter_v6 import adapter_class
from tools_v5 import RestrictedTools
from qwen_transport import LocalModel
from budget_v5 import atomic, BudgetStop


def evaluate(record, case):
    answer=record.get('answer') or {}
    if case['stage']=='submission':
        expected=case['expected_result_id']
        correct=answer.get('status')=='abstain' if expected is None else answer.get('status')=='success' and answer.get('result_id')==expected
        return {'correct':correct}
    if case['stage']=='navigation':
        recovered=any(c['name']=='get_observation' for e in record['trace'] for c in e.get('tool_calls',[]))
        correct=answer.get('status')=='success' and set(answer.get('fact_ids',[]))==set(case['expected_fact_ids'])
        return {'correct':correct and (recovered or not case['require_get_observation']), 'observation_recovered':recovered}
    from audit_rules import reference_bindings
    from audit_data import canonical_output
    from catalog_v5 import execute
    from score_v5 import score
    t=case['target']; refs=reference_bindings(t['calculator_id'],case['rows'],t['patient_id'],t['episode_id'])
    reference=execute(t['calculator_id'],{k:v['value'] for k,v in refs.items()})
    scored=score(record,{**t,'rows':case['rows'],'reference_output':reference,'canonical_benchmark':canonical_output(reference['value'],t['calculator_id'])})
    return {'correct':scored['trusted_correct'],**scored,'manual_transformation_review':'pending'}


async def episode(upstream,root,case,model):
    out=Path(root)/'episodes'/case['id'];out.mkdir(parents=True,exist_ok=False)
    visible={k:case[k] for k in ['id','stage','rows','target','prompt']}
    for k in ['seed_results','seed']:
        if k in case:visible[k]=case[k]
    if case.get('seed')=='unread_failure':visible['seed_unread_fact_ids']=case['expected_fact_ids']
    prompt=out/'prompt.txt';prompt.write_text(case['prompt']+'\nTarget scope: '+json.dumps(case['target']))
    args=upstream.parse_args(['--workspace',str(out),'--prompt-file',str(prompt),'--model','frozen-local',
        '--upstream','http://127.0.0.1:1/v1','--planning','off','--context-policy','none','--verification','off',
        '--max-tokens','4096','--context-window','32768','--timeout-seconds','300','--seed','42'])
    tools=RestrictedTools(visible['rows'],visible['target'])
    h=adapter_class(upstream)(args,'http://127.0.0.1:1/v1',out,upstream.EventLog(out),{},tools=tools,model=model,workflow=(),case=visible)
    await h.setup_tools(None)
    model.protocol.target_id=case['id'];model.protocol.episode_deadline=time.time()+300
    before=model.calls;error=None
    try:
        final=await h.build_graph().ainvoke({'turn':0,'termination':None,'final_text':None},config={'recursion_limit':100})
    except BaseException as exc:
        error=exc;final={'termination':'global_budget_stop' if isinstance(exc,BudgetStop) else 'runtime_failure'}
        model.protocol.budget.finish(status=final['termination'])
    trace=[]
    for event in h.history:
        m=event['message']
        trace.append({'role':'tool','name':event['tool_name'],'content':m.content,'status':m.status} if event['role']=='tool' else
                     {'role':'assistant','content':m.content,'tool_calls':m.tool_calls,'invalid_tool_calls':m.invalid_tool_calls})
    record={'id':case['id'],'stage':case['stage'],'termination':final.get('termination'),'answer':tools.answer,
            'model_requests':model.calls-before,'execution_state':h.execution.snapshot(),'result_ledger':h.execution.results,
            'seed_tool_requests':h.seed_tool_requests,'trace':trace,'time':time.time()}
    atomic(out/'RESULT.json',record);model.protocol.episode_deadline=None
    if error:raise error
    if final.get('termination') in ('provider_failure','protocol_error'):raise RuntimeError('Fatal provider/protocol failure; no restart')
    return record


def run(root,package,model_library,model_path):
    sys.path[:0]=[str(package),str(model_library)]
    from hcl.models import LocalTransformersModel
    from fixtures import cases
    root=Path(root);protocol=json.loads((root/'PROTOCOL.json').read_text())
    assert protocol['hcl_enabled'] is False and json.loads((root/'CPU_CHECK.json').read_text())['passed']
    backend=LocalTransformersModel(name='Qwen2.5-VL-7B-Instruct',path=str(model_path),torch_dtype='bfloat16',attn_implementation='sdpa',
        max_new_tokens=4096,answer_max_new_tokens=4096,temperature=0.,enable_thinking=False,local_files_only=True)
    model=LocalModel(backend,root);upstream=load_upstream(package);all_cases=cases();summaries=[];stage_results={};status='complete'
    try:
        for stage in ['submission','navigation','mapping']:
            current=[]
            for case in [c for c in all_cases if c['stage']==stage]:
                model.protocol.budget.check()
                atomic(root/'progress.json',{'stage':stage,'case':case['id'],'completed':len(summaries),'requests':model.calls,'time':time.time()})
                record=asyncio.run(episode(upstream,root,case,model))
                result=evaluate(record,case)
                summary={'id':case['id'],'stage':stage,'termination':record['termination'],'model_requests':record['model_requests'],
                         'invalid_tool_calls':sum(len(e.get('invalid_tool_calls',[])) for e in record['trace']),**result}
                summaries.append(summary);current.append(summary)
                atomic(root/'PARTIAL_METRICS.json',{'cases':summaries,'stages':stage_results})
            passed=len(current)==6 and all(r['correct'] for r in current)
            stage_results[stage]={'attempted':len(current),'correct':sum(r['correct'] for r in current),'passed':passed}
            if not passed:
                status='COMPONENT_GATE_FAIL';break
    except BudgetStop as exc:
        status='BUDGET_STOP';atomic(root/'STOP.json',{'reason':str(exc),'time':time.time()})
    except BaseException:
        status='RUNTIME_FAILURE';raise
    finally:
        atomic(root/'RUN_RESULT.json',{'status':status,'stages':stage_results,'cases':summaries,'model_requests':model.calls,
              'planned_cases':18,'attempted_cases':len(list((root/'episodes').glob('*/RESULT.json'))),
              'unattempted_cases':[c['id'] for c in all_cases if not (root/'episodes'/c['id']/'RESULT.json').exists()],
              'hcl_started':False,'synthetic_only':True,'time':time.time()})
    return 0 if status in ('complete','COMPONENT_GATE_FAIL') else 2
