"""Frozen real-controller integration. Predictions cannot access the gold map."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
import run_agent as rt
from qualified_tool import install_shared_guard
from policy import ARMS, COMMON, FINAL, FOCUSED, policy_context, summarize

ROOT = rt.ROOT
BUDGET = rt.BUDGET
PLAN = rt.PLAN
CONTEXT = {'phase': 'initializing'}
EVENTS = []
EPISODES = []
os.environ['TMPDIR'] = os.environ['R1136_TMP']
os.environ['TRITON_CACHE_DIR'] = str(Path(os.environ['R1136_CACHE'])/'triton')


def save(name, value):
    path = ROOT/name
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, default=str)+'\n')
    temp.replace(path)


def record(kind, value):
    row = {'time': time.time(), **CONTEXT, 'kind': kind, **value}
    EVENTS.append(row)
    with (ROOT/'private/events.jsonl').open('a') as stream:
        stream.write(json.dumps(row, default=str)+'\n')


def guard():
    if time.time() >= BUDGET['hard_deadline_epoch']:
        raise TimeoutError('Original deadline reached')
    if (ROOT/'STOPPED.json').exists() or (ROOT/'BUDGET_EXHAUSTED.json').exists():
        raise RuntimeError('Independent budget guard stopped')


rt.save, rt.record, rt.guard, rt.CONTEXT = save, record, guard, CONTEXT
from local_controller import LocalController
rt.SubscriptionModel = lambda: LocalController(PLAN, guard, record)


def load_four():
    from native_runtime import load_native
    wrappers, _, Agent, prompt = load_native(ROOT, record, guard)
    state = {'calls': 0, 'success': 0, 'images': set()}
    tools = [rt.ObservedTool(t.tool, state) for t in wrappers]
    return tools, state, Agent, prompt


def episode(item, arm, tools, state, Agent, prompt, bank, candidate):
    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.memory import MemorySaver
    guard()
    CONTEXT.update(phase='episode', episode_id=item['id'], arm=arm)
    before = len(EVENTS)
    context, retrieved = policy_context(arm, item['question'], bank, candidate)
    paths = item['images']
    state.update(calls=0, success=0, images={str(Path(p).resolve()) for p in paths})
    model = rt.SubscriptionModel()
    model.deadline = min(time.time()+PLAN['limits']['per_case_seconds'], BUDGET['hard_deadline_epoch'])
    log = ROOT/'outputs'/f'{item["id"]}_{arm}'
    log.mkdir(exist_ok=False)
    agent = Agent(model, tools, system_prompt=prompt+COMMON, checkpointer=MemorySaver(), log_dir=str(log))
    config = {'configurable': {'thread_id': '1'}, 'recursion_limit': 2*PLAN['limits']['per_case_controller_calls']+2}
    figures = ''.join(f'{label} located at {p}\n' for label, p in zip(item['image_labels'], paths))
    text = ('Answer this question correctly using chain of thought reasoning and carefully evaluating choices. '
            'Solve using our own vision and reasoning and thenuse tools to complement your reasoning. '
            'Trust your own judgement over any tools.\n'+item['question']+'\n'+figures+context)
    content = [{'type': 'text', 'text': text}] + [{'type': 'image_url', 'image_url': {'url': p}} for p in paths]
    started = time.time()
    result = {'id': item['id'], 'arm': arm, 'status': 'failed', 'predicted': None,
              'retrieved_ids': retrieved, 'reference_tier': item['reference_tier'], 'context_text': context}
    try:
        output = agent.workflow.invoke({'messages': [HumanMessage(content=content)]}, config)
        result['initial_final'] = output['messages'][-1].content
        output = agent.workflow.invoke({'messages': [HumanMessage(content=FINAL)]}, config)
        final = output['messages'][-1].content.strip()
        result.update(status='completed', final=final, predicted=final if final in list('ABCDEF') else None)
    except Exception as exc:
        result['error'] = repr(exc)
        record('episode_error', {'error': repr(exc), 'traceback': traceback.format_exc()})
    calls = [e for e in EVENTS[before:] if e['kind'] == 'tool_started']
    result.update(seconds=time.time()-started, controller_calls=model.calls, tool_calls=state['calls'],
                  successful_tools=state['success'], focused_requests=sum(e.get('name') == 'chest_xray_expert'
                  and e.get('args', {}).get('prompt') == FOCUSED for e in calls))
    EPISODES.append(result)
    save('private/episodes.json', EPISODES)
    save('PROGRESS.json', {'time': time.time(), 'recorded_episodes': len(EPISODES),
                          'planned_episodes': PLAN['planned_episodes'], 'context': CONTEXT,
                          'last_status': result['status']})


def main():
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(TimeoutError('Independent supervisor stop')))
    inputs = json.loads((ROOT/'private/inputs.json').read_text())
    bank = json.loads((ROOT/'private/raw_bank.json').read_text())
    candidate = json.loads((ROOT/'private/candidate.json').read_text())
    assert len(inputs) == PLAN['source_groups'] and len({x['id'] for x in inputs}) == len(inputs)
    assert len({x['source_key'] for x in inputs}) == len(inputs)
    assert len(inputs)*len(ARMS) == PLAN['planned_episodes']
    assert all(x['privacy_approved'] and all(Path(p).is_file() for p in x['images']) for x in inputs)
    save('STARTED.json', {'time': time.time(), 'original_start': BUDGET['started_epoch'],
                        'gpu': int(os.environ['CUDA_VISIBLE_DEVICES'])})
    if (ROOT/'private/episodes.json').exists():
        EPISODES[:] = json.loads((ROOT/'private/episodes.json').read_text())
    completed_keys = {(e['id'], e['arm']) for e in EPISODES}
    fatal = None
    try:
        available = {int(r.split(',')[0]): int(r.split(',')[1]) for r in subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], text=True).strip().splitlines()}
        gpu = int(os.environ['CUDA_VISIBLE_DEVICES'])
        save('GPU_PREFLIGHT.json', {'time': time.time(), 'gpu': gpu, 'free_mib': available[gpu]})
        if available[gpu] < 45000:
            raise RuntimeError('Insufficient GPU headroom; existing processes untouched')
        CONTEXT['phase'] = 'loading_native_tools'
        tools, state, Agent, prompt = load_four()
        install_shared_guard(tools, state, record)
        save('TOOLS_LOADED.json', {'time': time.time(), 'names': [t.name for t in tools], 'shared_visual_guard': True})
        CONTEXT['phase'] = 'preflight'
        image = inputs[0]['images'][0]
        state.update(calls=0, success=0, images={str(Path(image).resolve())})
        checks = []
        for tool in tools:
            args = {'image_paths': [image], 'prompt': 'Describe the chest X-ray findings.'} if tool.name == 'chest_xray_expert' else {'image_path': image}
            result = tool.invoke(args)
            status = result[1].get('analysis_status') if isinstance(result, tuple) else None
            checks.append({'name': tool.name, 'status': status})
            save('TOOL_PREFLIGHT.json', checks)
        if not all(x['status'] in {'completed', 'completed_no_finding'} for x in checks):
            raise RuntimeError('Native tool preflight failed')
        rt.SubscriptionModel().load()
        consecutive_failures = 0
        for i, item in enumerate(inputs):
            offset = item.get('execution_index', i) % len(ARMS)
            arms = ARMS[offset:] + ARMS[:offset]
            for arm in arms:
                if (item['id'], arm) in completed_keys:
                    continue
                episode(item, arm, tools, state, Agent, prompt, bank, candidate)
                latest = EPISODES[-1]
                format_failure = latest.get('error', '').startswith(('JSONDecodeError(', "ValueError('At most two tool calls"))
                infrastructure_failure = latest['status'] == 'failed' and not format_failure
                consecutive_failures = consecutive_failures + 1 if infrastructure_failure else 0
                if consecutive_failures >= 3:
                    raise RuntimeError('Three consecutive episode failures; preserve evidence for repair')
    except BaseException as exc:
        fatal = repr(exc)
        save('FAILED.json', {'time': time.time(), 'error': fatal, 'traceback': traceback.format_exc(), 'context': CONTEXT})
    finally:
        # Only the terminal evaluation path loads targets; they are never placed in controller jobs.
        gold = json.loads((ROOT/'private/sealed_gold.json').read_text())
        report = summarize(inputs, EPISODES, gold)
        report.update(fatal=fatal, finished_at=time.time(), seconds_since_original_start=time.time()-BUDGET['started_epoch'],
                      complete=fatal is None and len(EPISODES) == PLAN['planned_episodes']
                      and all(e['status'] == 'completed' and e['predicted'] is not None for e in EPISODES))
        save('AGGREGATE_REPORT.json', report)
        save('FINISHED.json', report)
    return 0 if fatal is None else 2


if __name__ == '__main__':
    raise SystemExit(main())
