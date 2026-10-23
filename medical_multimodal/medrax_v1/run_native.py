"""Frozen development qualification, using the upstream MedRAX graph unchanged."""
import json
import os
from pathlib import Path
import re
import time
import traceback

ROOT = Path(os.environ['RUN_ROOT']).resolve()


def save(name, value):
    path = ROOT / name
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def score_letter(text):
    text = text.strip()
    if re.fullmatch(r'[A-F][).]?', text):
        return text[0]
    found = re.findall(r'(?:^|\n)\s*(?:Final\s+)?Answer:\s*([A-F])(?:[).]|\s|$)', text, re.I)
    return found[-1].upper() if found else None


def main():
    from langchain_core.messages import HumanMessage
    from native_runtime import load_native, NoUpdateHCL
    from vision_transport import LocalVisionModel
    plan = json.loads((ROOT / 'code/native_protocol.json').read_text())
    (ROOT / 'outputs').mkdir(exist_ok=True)
    records, results, preflight = [], [], []
    context = {'phase': 'loading', 'episode': None}

    def record(kind, value):
        item = {'time': time.time(), **context, 'kind': kind, **value}
        records.append(item)
        with (ROOT / 'native_events.private.jsonl').open('a') as stream:
            stream.write(json.dumps(item, default=str) + '\n')

    def guard():
        data = json.loads((ROOT / 'guard.json').read_text())
        if data['stop'] or time.time() - data['time'] > 90:
            raise RuntimeError('Resource guard stale or stopped')

    all_rows = [json.loads(x) for x in (ROOT / 'data/chestagentbench/metadata.jsonl').read_text().splitlines()]
    # The runner gives only questions/images to the agent; gold scoring happens after execution.
    rows = {x['full_question_id']: {k: x[k] for k in ['question', 'images', 'case_id']} for x in all_rows}
    selected = plan['panel']
    save('NATIVE_STARTED.json', {'time': time.time(), 'panel_count': len(selected), 'hcl_updates': False})
    fatal = None
    try:
        tools, state, agent_cls, prompt = load_native(ROOT, record, guard)
        images_dir = ROOT / 'data/chestagentbench/images'
        first_image = str((images_dir / selected[0]['images'][0]).resolve())
        state['images'] = {first_image}
        context.update(phase='direct_tool_preflight', episode=selected[0]['full_question_id'])
        for tool in tools:
            args = {'image_paths': [first_image], 'prompt': 'Describe the chest X-ray findings.',
                    'max_new_tokens': 128} if tool.name == 'chest_xray_expert' else {'image_path': first_image}
            value = tool.invoke(args)
            ok = isinstance(value, tuple) and value[1].get('analysis_status') == 'completed'
            preflight.append({'tool': tool.name, 'success': ok})
        if not all(x['success'] for x in preflight):
            raise RuntimeError('Native tool preflight failed; planner panel not started')
        config = json.loads((ROOT / 'runtime.private.json').read_text())
        context.update(phase='planner_loading', episode=None)
        model = LocalVisionModel(config['planner_path'], plan['limits'], record, guard)
        for item in selected:
            guard()
            qid = item['full_question_id']
            row = rows[qid]
            context.update(phase='native_panel', episode=qid)
            paths = [str((images_dir / name).resolve()) for name in row['images']]
            state.update(calls=0, max_calls=plan['limits']['tool_calls'], images=set(paths))
            model.start_episode()
            begin = time.time()
            log_dir = ROOT / 'outputs' / qid
            log_dir.mkdir(exist_ok=True)
            graph = agent_cls(NoUpdateHCL(model), tools, system_prompt=prompt, log_dir=str(log_dir))
            message = HumanMessage(content=[{'type': 'image', 'image': p} for p in paths] + [{
                'type': 'text', 'text': row['question'] + '\n\nAvailable image paths:\n' +
                    '\n'.join(paths) + '\nGive your final choice as Answer: <letter>.'}])
            result = {'id': qid, 'case_id': row['case_id'], 'status': 'failed', 'predicted': None}
            try:
                output = graph.workflow.invoke({'messages': [message]}, {'recursion_limit': 12})
                final = output['messages'][-1]
                result.update(status='completed', predicted=score_letter(final.content), final=final.content)
            except Exception as exc:
                result['error'] = repr(exc)
                record('episode_error', {'error': repr(exc), 'traceback': traceback.format_exc()})
            result.update(seconds=time.time() - begin, planner_calls=model.calls,
                          planner_output_tokens=model.tokens, tool_calls=state['calls'])
            results.append(result)
            save('native_results.private.json', results)
    except Exception as exc:
        fatal = repr(exc)
        record('fatal', {'error': fatal, 'traceback': traceback.format_exc()})
    completed_ids = {x['id'] for x in results}
    for item in selected:
        if item['full_question_id'] not in completed_ids:
            results.append({'id': item['full_question_id'], 'case_id': item['case_id'],
                            'status': 'not_run', 'predicted': None, 'reason': fatal or 'phase ended'})
    gold = {x['full_question_id']: x['answer'] for x in all_rows}
    for result in results:
        raw_gold = str(gold[result['id']]).strip()
        letter = score_letter(raw_gold) or (raw_gold[0] if re.match(r'^[A-F][).\s]', raw_gold) else None)
        if letter is None:
            raise ValueError('Unsupported gold format; do not invent a score')
        result['correct'] = result['predicted'] == letter
    successful_episodes = {r['episode'] for r in records if r['phase'] == 'native_panel' and
                           r['kind'] == 'tool_finished' and r['status'] == 'completed'}
    starts = [r for r in records if r['kind'] == 'planner_started']
    summary = {'time': time.time(), 'stage': plan['stage'], 'denominator': len(selected),
        'attempted': sum(x['status'] != 'not_run' for x in results),
        'completed': sum(x['status'] == 'completed' for x in results),
        'correct': sum(x['correct'] for x in results),
        'valid_final': sum(x['predicted'] is not None for x in results),
        'episodes_with_successful_tool': len(successful_episodes),
        'planner_calls': len(starts),
        'image_payload_verified': bool(starts) and all(x['images'] > 0 for x in starts),
        'direct_tool_preflight': preflight,
        'expert_generations': sum(x['kind'] == 'expert_generation_started' for x in records),
        'fatal': fatal, 'hcl_updates': 0, 'scientific_efficacy_claim': False}
    gate = plan['qualification_gate']
    summary['native_qualified'] = (fatal is None and len(preflight) == 4 and all(x['success'] for x in preflight)
        and summary['image_payload_verified'] and summary['correct'] >= gate['minimum_correct_of_12']
        and summary['valid_final'] >= gate['minimum_valid_final_of_12']
        and len(successful_episodes) >= gate['minimum_episodes_with_successful_tool'])
    save('native_results.private.json', results)
    save('NATIVE_FINISHED.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary['native_qualified'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
