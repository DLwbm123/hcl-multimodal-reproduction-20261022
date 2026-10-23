"""One predeclared prompt contrast on fresh development cases; no HCL updates."""
import json
import os
from pathlib import Path
import sys
import time
import traceback

# Local checkout uses the sibling runtime; remote launch copies that pinned source beside this file.
sys.path.append(str(Path(__file__).resolve().parents[1] / 'medrax_v1'))
from run_native import score_letter, save
from native_runtime import load_native, NoUpdateHCL
from vision_transport import LocalVisionModel

ROOT = Path(os.environ['RUN_ROOT']).resolve()


def main():
    from langchain_core.messages import HumanMessage
    plan = json.loads((ROOT / 'code/protocol.json').read_text())
    config = json.loads((ROOT / 'runtime.private.json').read_text())
    all_rows = [json.loads(x) for x in (ROOT / 'data/chestagentbench/metadata.jsonl').read_text().splitlines()]
    questions = {x['full_question_id']: {k: x[k] for k in ['question', 'images']} for x in all_rows}
    context = {'phase': 'loading', 'episode': None, 'arm': None}
    events, results = [], []
    (ROOT / 'outputs').mkdir(exist_ok=True)

    def record(kind, value):
        item = {'time': time.time(), **context, 'kind': kind, **value}
        events.append(item)
        with (ROOT / 'events.private.jsonl').open('a') as stream:
            stream.write(json.dumps(item, default=str) + '\n')

    def guard():
        value = json.loads((ROOT / 'guard.json').read_text())
        if value['stop'] or time.time() - value['time'] > 90:
            raise RuntimeError('Time/storage guard stopped or stale')

    save('STARTED.json', {'time': time.time(), 'planned_episodes': 24, 'hcl_updates': 0})
    fatal = None
    try:
        tools, state, agent_cls, prompt = load_native(ROOT, record, guard)
        model = LocalVisionModel(config['planner_path'], plan['limits'], record, guard)
        for index, item in enumerate(plan['panel']):
            qid = item['full_question_id']
            row = questions[qid]
            paths = [str((ROOT / 'data/chestagentbench/images' / name).resolve()) for name in row['images']]
            order = ['A_native', 'B_consult'] if index % 2 == 0 else ['B_consult', 'A_native']
            for arm in order:
                guard()
                context.update(phase='paired_panel', episode=qid, arm=arm)
                model.start_episode()
                state.update(calls=0, max_calls=plan['limits']['tool_calls'], images=set(paths))
                begin = time.time()
                directory = ROOT / 'outputs' / (qid + '_' + arm)
                directory.mkdir(exist_ok=True)
                agent = agent_cls(NoUpdateHCL(model), tools, system_prompt=prompt, log_dir=str(directory))
                text = row['question'] + '\n\nAvailable image paths:\n' + '\n'.join(paths)
                if plan['arms'][arm]:
                    text += '\n\n' + plan['arms'][arm]
                text += '\nGive your final choice as Answer: <letter>.'
                message = HumanMessage(content=[{'type': 'image', 'image': p} for p in paths] +
                                       [{'type': 'text', 'text': text}])
                result = {'id': qid, 'case_id': item['case_id'], 'arm': arm,
                          'status': 'failed', 'predicted': None}
                try:
                    output = agent.workflow.invoke({'messages': [message]}, {'recursion_limit': 12})
                    final = output['messages'][-1].content
                    result.update(status='completed', predicted=score_letter(final), final=final)
                except Exception as exc:
                    result['error'] = repr(exc)
                    record('episode_error', {'error': repr(exc), 'traceback': traceback.format_exc()})
                result.update(seconds=time.time() - begin, planner_calls=model.calls,
                              planner_output_tokens=model.tokens, tool_calls=state['calls'])
                results.append(result)
                save('results.private.json', results)
    except Exception as exc:
        fatal = repr(exc)
        record('fatal', {'error': fatal, 'traceback': traceback.format_exc()})
    completed = {(x['id'], x['arm']) for x in results}
    for item in plan['panel']:
        for arm in plan['arms']:
            if (item['full_question_id'], arm) not in completed:
                results.append({'id': item['full_question_id'], 'case_id': item['case_id'], 'arm': arm,
                                'status': 'not_run', 'predicted': None, 'reason': fatal})
    gold = {x['full_question_id']: x['answer'].strip() for x in all_rows}
    for result in results:
        assert gold[result['id']] in 'ABCDEF' and len(gold[result['id']]) == 1
        result['correct'] = result['predicted'] == gold[result['id']]
    summary = {'time': time.time(), 'denominator': 24, 'arms': {}, 'fatal': fatal,
               'hcl_updates': 0, 'scientific_efficacy_claim': False}
    for arm in plan['arms']:
        rows = [x for x in results if x['arm'] == arm]
        observations = [x for x in events if x['arm'] == arm]
        starts = [x for x in observations if x['kind'] == 'planner_started']
        successful = {x['episode'] for x in observations if x['kind'] == 'tool_finished' and x['status'] == 'completed'}
        summary['arms'][arm] = {
            'denominator': 12, 'attempted': sum(x['status'] != 'not_run' for x in rows),
            'completed': sum(x['status'] == 'completed' for x in rows),
            'correct': sum(x['correct'] for x in rows),
            'valid_final': sum(x['predicted'] is not None for x in rows),
            'successful_tool_episodes': len(successful),
            'runtime_error_episodes': sum(x['status'] == 'failed' for x in rows),
            'image_payload_verified': bool(starts) and all(x['images'] > 0 for x in starts),
            'planner_calls': len(starts),
            'planner_input_tokens': sum(x['input_tokens'] for x in starts),
            'planner_output_tokens': sum(x['output_tokens'] for x in observations if x['kind'] == 'planner_finished'),
            'expert_generations': sum(x['kind'] == 'expert_generation_started' for x in observations)}
    a, b = summary['arms']['A_native'], summary['arms']['B_consult']
    a_ok = a['correct'] >= 3 and a['valid_final'] >= 4 and a['successful_tool_episodes'] >= 3 and a['image_payload_verified']
    b_ok = b['correct'] >= max(3, a['correct']) and b['valid_final'] >= 4 and b['successful_tool_episodes'] >= 6 and b['runtime_error_episodes'] == 0 and b['image_payload_verified']
    summary['selected_shared_base'] = ('A_native' if a_ok else 'B_consult' if b_ok else None) if fatal is None else None
    summary['paired_correctness'] = {label: sum(test(next(x for x in results if x['id'] == item['full_question_id'] and x['arm'] == 'A_native')['correct'],
        next(x for x in results if x['id'] == item['full_question_id'] and x['arm'] == 'B_consult')['correct']) for item in plan['panel'])
        for label, test in {'both_correct': lambda a, b: a and b, 'A_only': lambda a, b: a and not b,
                            'B_only': lambda a, b: b and not a, 'both_wrong': lambda a, b: not a and not b}.items()}
    save('results.private.json', results)
    save('FINISHED.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary['selected_shared_base'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
