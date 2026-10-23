"""Paired raw-input ablation with independent mapping probes; no score gates."""
import asyncio
import json
import sys
import time
from pathlib import Path
import bootstrap_v7
from adapter_v5 import load_upstream
from adapter_v7 import adapter_class
from tools_v5 import RestrictedTools
from qwen_transport import LocalModel
from budget_v5 import atomic, BudgetStop


async def episode(upstream,root,case,model):
    out=Path(root)/'episodes'/case['id'];out.mkdir(parents=True,exist_ok=False)
    visible={k:case[k] for k in ['id','stage','rows','target','prompt','arm']}
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
    record={'id':case['id'],'stage':case['stage'],'case_id':case['case_id'],'arm':case['arm'],'termination':final.get('termination'),'answer':tools.answer,
            'model_requests':model.calls-before,'execution_state':h.execution.snapshot(),'result_ledger':h.execution.results,
            'seed_tool_requests':h.seed_tool_requests,'programmatic_reads':h.programmatic_reads,
            'auto_read_bytes':h.auto_read_bytes,'initial_execution_state':h.initial_execution_state,'trace':trace,'time':time.time()}
    atomic(out/'RESULT.json',record);model.protocol.episode_deadline=None
    if error:raise error
    if final.get('termination') in ('provider_failure','protocol_error'):raise RuntimeError('Fatal provider/protocol failure; no restart')
    return record


def run(root,package,model_library,model_path):
    sys.path[:0] = [str(package),str(model_library)]
    from hcl.models import LocalTransformersModel
    from fixtures_v7 import schedule
    from evaluate_v7 import evaluate
    root = Path(root)
    protocol = json.loads((root/'PROTOCOL.json').read_text())
    assert protocol['hcl_enabled'] is False and protocol['frozen_epoch']
    assert json.loads((root/'CPU_CHECK.json').read_text())['passed']
    backend = LocalTransformersModel(name='Qwen2.5-VL-7B-Instruct',path=str(model_path),
        torch_dtype='bfloat16',attn_implementation='sdpa',max_new_tokens=4096,
        answer_max_new_tokens=4096,temperature=0.,enable_thinking=False,local_files_only=True)
    model = LocalModel(backend,root)
    upstream = load_upstream(package)
    cases = schedule()
    summaries = []
    status = 'COMPLETE_PENDING_AUDIT'
    try:
        for case in cases:
            model.protocol.budget.check()
            atomic(root/'progress.json',{'case':case['id'],'completed':len(summaries),
                'planned':len(cases),'requests':model.calls,'time':time.time()})
            record = asyncio.run(episode(upstream,root,case,model))
            # Evaluator results are saved separately, never passed to later episodes.
            result = evaluate(record,case)
            summaries.append({'id':case['id'],'case_id':case['case_id'],'arm':case['arm'],
                'stage':case['stage'],'termination':record['termination'],
                'model_requests':record['model_requests'],
                'programmatic_read_calls':len(record['programmatic_reads']),
                'programmatic_read_bytes':record['auto_read_bytes'],
                'invalid_tool_calls':sum(len(e.get('invalid_tool_calls',[])) for e in record['trace']),**result})
            atomic(root/'PARTIAL_METRICS.json',{'cases':summaries})
    except BudgetStop as exc:
        status = 'BUDGET_STOP'
        atomic(root/'STOP.json',{'reason':str(exc),'time':time.time()})
    except BaseException:
        status = 'RUNTIME_FAILURE'
        raise
    finally:
        atomic(root/'RUN_RESULT.json',{'status':status,'cases':summaries,
            'model_requests':model.calls,'planned_cases':len(cases),
            'attempted_cases':len(list((root/'episodes').glob('*/RESULT.json'))),
            'unattempted_cases':[c['id'] for c in cases if not (root/'episodes'/c['id']/'RESULT.json').exists()],
            'hcl_started':False,'synthetic_only':True,'time':time.time()})
    return 0 if status=='COMPLETE_PENDING_AUDIT' else 2
