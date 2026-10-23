"""One frozen H0 pass. No proposer, HCL entry, retries or reselection."""
import asyncio
import json
import time
from pathlib import Path

from adapter_v5 import load_upstream, adapter_class
from tools_v5 import RestrictedTools
from qwen_transport import LocalModel
from budget_v5 import BudgetStop, atomic


async def episode(upstream, root, case, model):
    out = Path(root)/case['id']
    out.mkdir(parents=True, exist_ok=False)
    prompt = out/'prompt.txt'; prompt.write_text(case['prompt'])
    args = upstream.parse_args(['--workspace', str(out), '--prompt-file', str(prompt), '--model', 'frozen-local',
        '--upstream', 'http://127.0.0.1:1/v1', '--planning', 'off', '--context-policy', 'none',
        '--verification', 'off', '--max-tokens', '4096', '--context-window', '32768', '--timeout-seconds', '900', '--seed', '42'])
    tools = RestrictedTools(case['rows'], case['target'])
    h = adapter_class(upstream)(args, 'http://127.0.0.1:1/v1', out, upstream.EventLog(out), {}, tools=tools, model=model, workflow=())
    await h.setup_tools(None)
    model.protocol.target_id = case['id']
    model.protocol.episode_deadline = time.time()+900
    before = model.calls
    error = None
    try:
        final = await h.build_graph().ainvoke({'turn': 0, 'termination': None, 'final_text': None}, config={'recursion_limit': 110})
    except BaseException as exc:
        error = exc
        final = {'termination': 'global_budget_stop' if isinstance(exc, BudgetStop) else 'runtime_failure'}
        model.protocol.budget.finish(status=final['termination'])
    trace = []
    for event in h.history:
        m = event['message']
        if event['role'] == 'tool':
            trace.append({'role': 'tool', 'name': event['tool_name'], 'content': m.content, 'status': m.status})
        else:
            trace.append({'role': 'assistant', 'content': m.content, 'tool_calls': m.tool_calls, 'invalid_tool_calls': m.invalid_tool_calls})
    record = {'id': case['id'], 'termination': final.get('termination'), 'answer': tools.answer,
              'model_requests': model.calls-before, 'execution_state': h.execution.snapshot(),
              'result_ledger': h.execution.results, 'trace': trace, 'time': time.time()}
    atomic(out/'RESULT.private.json', record)
    model.protocol.episode_deadline = None
    if error:
        raise error
    if final.get('termination') in ('provider_failure', 'protocol_error'):
        raise RuntimeError('Unclassified provider failure; no retry')
    return record


def run(root, package, model_library, model_path):
    import sys
    sys.path[:0] = [str(package), str(model_library)]
    from hcl.models import LocalTransformersModel
    root = Path(root)
    protocol = json.loads((root/'PROTOCOL_H0_V5.json').read_text())
    assert protocol['hcl_enabled'] is False and protocol['rollouts'] == 18
    assert json.loads((root/'CPU_CHECK.json').read_text())['passed']
    visible = json.loads((root/'private'/'MODEL_INPUTS.private.json').read_text())
    assert len(visible) == 18 and len({c['id'] for c in visible}) == 18
    assert all(set(c) == {'id', 'rows', 'prompt', 'target'} for c in visible)
    backend = LocalTransformersModel(name='Qwen2.5-VL-7B-Instruct', path=str(model_path), torch_dtype='bfloat16',
        attn_implementation='sdpa', max_new_tokens=4096, answer_max_new_tokens=4096,
        temperature=0., enable_thinking=False, local_files_only=True)
    model = LocalModel(backend, root)
    upstream = load_upstream(package)
    status = 'complete'
    try:
        for ordinal, case in enumerate(visible):
            model.protocol.budget.check()
            atomic(root/'progress.json', {'ordinal': ordinal, 'total': 18, 'target': case['id'],
                                         'requests': model.calls, 'time': time.time()})
            asyncio.run(episode(upstream, root/'episodes', case, model))
    except BudgetStop as exc:
        status = 'BUDGET_STOP'
        atomic(root/'STOP.json', {'status': status, 'reason': str(exc), 'time': time.time()})
    except BaseException:
        status = 'RUNTIME_FAILURE'
        raise
    finally:
        atomic(root/'RUN_RESULT.json', {'status': status, 'model_requests': model.calls,
            'completed_records': len(list((root/'episodes').glob('*/RESULT.private.json'))),
            'planned_targets': 18, 'hcl_started': False, 'time': time.time()})
    return 0 if status == 'complete' else 2
