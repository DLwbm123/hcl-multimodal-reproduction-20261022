"""CPU native-graph fixture: seed correlation, no oracle, full failure denominator."""
import json
import os
from pathlib import Path
import tempfile
import time


def main():
    from langchain_core.messages import AIMessage
    source = Path(os.environ['RUN_ROOT'])
    plan = json.loads((source / 'code/protocol.json').read_text())
    with tempfile.TemporaryDirectory(dir=source / 'tmp') as folder:
        root = Path(folder)
        os.environ['RUN_ROOT'] = str(root)
        import run_paired as runner
        from native_runtime import load_module
        from vision_transport import serialize
        Agent = load_module(source / 'upstream/medrax/agent/agent.py', 'fixture_agent').Agent
        for path in ['code', 'canonical', 'data/chestagentbench']:
            (root / path).mkdir(parents=True, exist_ok=True)
        (root / 'code/protocol.json').write_text(json.dumps(plan))
        (root / 'runtime.private.json').write_text(json.dumps({'planner_path': 'unused'}))
        (root / 'guard.json').write_text(json.dumps({'time': time.time(), 'stop': False}))
        rows = []
        for index, item in enumerate(plan['panel']):
            rows.append({'full_question_id': item['full_question_id'], 'images': item['images'],
                         'question': f'Fixture question {index}', 'answer': 'F',
                         'explanation': 'SECRET_EXPLANATION'})
            (root / 'canonical' / (item['full_question_id'] + '.png')).touch()
        (root / 'data/chestagentbench/metadata.jsonl').write_text('\n'.join(map(json.dumps, rows)))
        state, observed = {}, []
        class Model:
            def __init__(self, path, limits, record, guard):
                self.record = record
            def start_episode(self):
                self.calls = self.tokens = 0
            def bind_tools(self, tools):
                return self
            def invoke(self, messages):
                converted = serialize(messages, [])  # actual call/result correlation check
                raw = json.dumps(converted)
                assert 'SECRET_EXPLANATION' not in raw and '"answer"' not in raw
                assert messages[1].content[0]['type'] == 'image'
                observed.append(messages)
                self.calls = 1
                self.record('planner_started', {'images': 1, 'input_tokens': 1})
                if 'Fixture question 3\n' in raw.replace('\\n', '\n') and len(messages) == 4 and messages[-1].name == 'chest_xray_expert':
                    raise RuntimeError('Injected failure after real native seed execution')
                self.record('planner_finished', {'output_tokens': 1})
                return AIMessage(content='Answer: F')
        class Tool:
            def __init__(self, name, record):
                self.name, self.record = name, record
            def invoke(self, args):
                state['calls'] += 1
                assert state['calls'] <= 8
                paths = args.get('image_paths', [args.get('image_path')])
                assert set(paths) == state['images']
                if self.name == 'chest_xray_expert':
                    assert args['prompt'].startswith('Fixture question') and args['max_new_tokens'] == 512
                self.record('tool_finished', {'name': self.name, 'status': 'completed'})
                return ({'fixture_evidence': 'unscored'}, {'analysis_status': 'completed'})
        def load(root, record, guard):
            return [Tool(x, record) for x in ['chest_xray_classifier', 'chest_xray_expert']], state, Agent, 'Identical system'
        runner.LocalVisionModel, runner.load_native = Model, load
        assert runner.main() == 0
        summary = json.loads((root / 'FINISHED.json').read_text())
        results = json.loads((root / 'results.private.json').read_text())
        assert len(results) == len(observed) == summary['denominator'] == 36
        assert summary['arms']['C_expert']['runtime_error_episodes'] == 1
        assert summary['arms']['C_expert']['correct'] == 11
        assert summary['arms']['A_native']['successful_seed_episodes'] == 0
        for arm in ['B_classifier', 'C_expert']:
            assert summary['arms'][arm]['successful_seed_episodes'] == 12
            assert summary['arms'][arm]['autonomous_tool_episodes'] == 0
        for index in range(0, 36, 3):
            batch = observed[index:index + 3]
            assert batch[0][1].content == batch[1][1].content == batch[2][1].content
            assert sorted(map(len, batch)) == [2, 4, 4]
        assert not summary['autonomous_qualification']
        print('CAUSAL_DESIGN_OK: native graph, seed/result correlation, 36 denominator, injected failure retained; model_generations=0')


if __name__ == '__main__':
    main()
