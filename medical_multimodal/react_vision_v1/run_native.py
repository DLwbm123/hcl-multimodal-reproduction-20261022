"""Native Lagent ReAct with tool-mediated vision; no updates or answer oracle."""
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time
import traceback

ROOT = Path(os.environ['RUN_ROOT'])
sys.path.append(str(Path(__file__).resolve().parents[1] / 'medrax_v1'))
from vision_transport import LocalVisionModel


def save(name, value):
    path = ROOT / name
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temp.replace(path)


class TextController(LocalVisionModel):
    """Reuse model loading/budgets; original ReAct controller receives text only."""
    def chat(self, messages, **kwargs):
        import torch
        self.guard()
        if self.calls >= self.limits['planner_calls'] or time.time() >= self.deadline:
            raise RuntimeError('Planner/time budget exhausted')
        inputs = self.processor.tokenizer.apply_chat_template(messages, tokenize=True,
            add_generation_prompt=True, return_tensors='pt').to(self.model.device)
        if inputs.shape[-1] > self.limits['input_tokens']:
            raise RuntimeError('Context budget exceeded')
        allowed = min(self.limits['output_tokens_per_call'], self.limits['output_tokens_total'] - self.tokens)
        if allowed <= 0:
            raise RuntimeError('Output budget exhausted')
        self.calls += 1
        start = time.time()
        self.record('planner_started', {'input_tokens': int(inputs.shape[-1]), 'controller_images': 0})
        with torch.inference_mode():
            output = self.model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                do_sample=False, max_new_tokens=allowed, max_time=max(1, min(180, self.deadline - start)))
        tokens = output[0, inputs.shape[-1]:]
        self.tokens += len(tokens)
        raw = self.processor.tokenizer.decode(tokens, skip_special_tokens=True)
        self.record('planner_finished', {'output_tokens': len(tokens), 'seconds': time.time() - start, 'raw': raw})
        return raw


def main():
    from lagent import ReAct, ActionExecutor
    from lagent.actions import BaseAction
    from lagent.schema import ActionReturn, ActionStatusCode
    from PIL import Image
    import re
    plan = json.loads((ROOT / 'code/protocol.json').read_text())
    config = json.loads((ROOT / 'runtime.private.json').read_text())
    data = [json.loads(line) for line in (ROOT / 'data/chestagentbench/metadata.jsonl').read_text().splitlines()]
    visible = {x['full_question_id']: x['question'] for x in data}
    context = {'phase': 'loading', 'episode': None}
    events, results, preflight = [], [], []
    state = {'calls': 0, 'image': None}
    transport = {'failed': False, 'sequence': 0}
    def record(kind, value):
        row = {'time': time.time(), **context, 'kind': kind, **value}
        events.append(row)
        with (ROOT / 'events.private.jsonl').open('a') as stream:
            stream.write(json.dumps(row, default=str) + '\n')
    def guard():
        if transport['failed']:
            raise RuntimeError('Tool transport failed; no reuse of pending responses')
        value = json.loads((ROOT / 'guard.json').read_text())
        if value['stop'] or time.time() - value['time'] > 90:
            raise RuntimeError('Time/storage guard stopped or stale')
    def receive(worker, timeout):
        if not select.select([worker.stdout], [], [], timeout)[0]:
            raise TimeoutError('Tool worker response deadline')
        line = worker.stdout.readline()
        if not line:
            raise RuntimeError('Tool worker exited')
        return json.loads(line)
    worker = None
    fatal = None
    save('STARTED.json', {'time': time.time(), 'planned_episodes': 12, 'hcl_updates': 0})
    try:
        worker = subprocess.Popen(['./bin/python', '/data/bmw/workers/r943.py'], cwd=config['tool_environment'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=(ROOT / 'tools.log').open('a'), text=True, bufsize=1)
        save('TOOL_LAUNCH.json', {'pid': worker.pid, 'time': time.time(), 'shares_worker_process_group': True})
        ready = receive(worker, 600)
        save('TOOLS_READY.json', ready)
        if not ready.get('ready'):
            raise RuntimeError(ready)
        def call(name, args):
            guard()
            state['calls'] += 1
            if state['calls'] > plan['limits']['tool_calls']:
                raise RuntimeError('Tool budget exhausted')
            transport['sequence'] += 1
            request_id = transport['sequence']
            record('tool_started', {'name': name, 'args': args, 'request_id': request_id})
            worker.stdin.write(json.dumps({'request_id': request_id, 'name': name, 'args': args, 'allowed_image': state['image']}) + '\n')
            worker.stdin.flush()
            try:
                result = receive(worker, min(180, max(1, state.get('deadline', time.time()+180) - time.time())))
                if result['request_id'] != request_id:
                    raise RuntimeError('Tool response identity mismatch')
            except Exception:
                transport['failed'] = True
                raise
            record('tool_finished', {'name': name, **result})
            return result
        context['phase'] = 'direct_preflight'
        first = plan['panel'][0]['full_question_id']
        path = str(ROOT / 'canonical' / (first + '.png'))
        state.update(calls=0, image=path)
        with Image.open(path) as image:
            width, height = image.size
        for name, args in [('ImageDescription', {'image': path}), ('RegionAttributeDescription',
                {'image': path, 'bbox': str((width//4, height//4, 3*width//4, 3*height//4)), 'attribute': 'visible findings'}),
                ('OCR', {'image': path}), ('Calculator', {'expression': '2+3'})]:
            preflight.append({'name': name, **call(name, args)})
            save('PREFLIGHT.private.json', preflight)
        if not all(x['ok'] for x in preflight):
            raise RuntimeError('Direct native tools failed; no qualification episodes launched')
        model = TextController(config['planner_path'], plan['limits'], record, guard)
        metadata = json.loads((ROOT / 'code/toolmeta.json').read_text())
        class RemoteAction(BaseAction):
            def __init__(self, name):
                meta = metadata[name]
                params = [{'name': p['name'], 'description': p['description'], 'type': 'STRING'} for p in meta['inputs']]
                super().__init__(description={'name': name, 'description': meta['description'],
                    'parameters': params, 'required': [p['name'] for p in meta['inputs'] if not p['optional']]})
            def run(self, **kwargs):
                result = call(self.name, kwargs)
                return ActionReturn(type=self.name, args=kwargs,
                    result=[{'type': 'text', 'content': result.get('result', '')}] if result['ok'] else None,
                    errmsg=result.get('error'), state=ActionStatusCode.SUCCESS if result['ok'] else ActionStatusCode.API_ERROR)
        for item in plan['panel']:
            guard()
            qid = item['full_question_id']
            context.update(phase='native_panel', episode=qid)
            model.start_episode()
            state.update(calls=0, image=str(ROOT / 'canonical' / (qid + '.png')), deadline=model.deadline)
            agent = ReAct(model, ActionExecutor([RemoteAction(n) for n in plan['tool_catalog']]), max_turn=plan['limits']['planner_calls'])
            row = {'id': qid, 'case_id': item['case_id'], 'status': 'failed', 'predicted': None}
            start = time.time()
            try:
                output = agent.chat([{'role': 'user', 'content': 'The related image file is at `' + state['image'] + '`.'},
                    {'role': 'user', 'content': visible[qid] + '\nGive your final choice as Answer: <letter>.'}])
                final = output.response
                matches = re.findall(r'\bAnswer\s*:\s*([A-F])\b', final, re.I)
                predicted = matches[-1].upper() if matches else final.strip() if re.fullmatch(r'[A-F]', final.strip()) else None
                row.update(status='completed', predicted=predicted, final=final, actions=output.actions)
            except Exception as exc:
                row['error'] = repr(exc)
                record('episode_error', {'error': repr(exc), 'traceback': traceback.format_exc()})
            row.update(seconds=time.time()-start, planner_calls=model.calls, tool_calls=state['calls'])
            results.append(row)
            save('results.private.json', results)
    except Exception as exc:
        fatal = repr(exc)
        record('fatal', {'error': fatal, 'traceback': traceback.format_exc()})
    finally:
        if worker is not None:
            try:
                worker.stdin.write('null\n'); worker.stdin.flush(); worker.wait(timeout=10)
            except Exception:
                worker.terminate()
                try: worker.wait(timeout=10)
                except subprocess.TimeoutExpired: worker.kill(); worker.wait()
    done = {x['id'] for x in results}
    for item in plan['panel']:
        if item['full_question_id'] not in done:
            results.append({'id': item['full_question_id'], 'case_id': item['case_id'], 'status': 'not_run', 'predicted': None, 'reason': fatal})
    gold = {x['full_question_id']: x['answer'].strip() for x in data}
    for row in results: row['correct'] = row['predicted'] == gold[row['id']]
    observations = [x for x in events if x['phase'] == 'native_panel']
    visual = {x['episode'] for x in observations if x['kind']=='tool_finished' and x['ok'] and x['name'] in ['ImageDescription','RegionAttributeDescription','OCR']}
    summary = {'time': time.time(), 'denominator': 12, 'correct': sum(x['correct'] for x in results),
        'completed': sum(x['status']=='completed' for x in results), 'valid_final': sum(x['predicted'] is not None for x in results),
        'successful_visual_tool_episodes': len(visual), 'direct_tools_passed': len(preflight)==4 and all(x['ok'] for x in preflight),
        'planner_calls': sum(x['kind']=='planner_started' for x in observations),
        'planner_input_tokens': sum(x.get('input_tokens',0) for x in observations if x['kind']=='planner_started'),
        'planner_output_tokens': sum(x.get('output_tokens',0) for x in observations if x['kind']=='planner_finished'),
        'vision_generations': sum(len(x.get('generations',[])) for x in observations if x['kind']=='tool_finished'),
        'fatal': fatal, 'hcl_updates': 0, 'controller_has_direct_image_payload': False,
        'external_medcta_cases_used': 0, 'scientific_efficacy_claim': False}
    summary['gate_pass'] = (fatal is None and summary['direct_tools_passed'] and summary['correct']>=3
        and summary['valid_final']>=6 and len(visual)>=6)
    save('results.private.json', results); save('FINISHED.json', summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary['gate_pass'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
