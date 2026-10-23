"""CPU-only check of paired prompt isolation and failure accounting; no model calls."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace


def main():
    source_root = Path(os.environ['RUN_ROOT'])
    plan = json.loads((source_root / 'code/protocol.json').read_text())
    with tempfile.TemporaryDirectory(dir=source_root / 'tmp') as folder:
        root = Path(folder)
        os.environ['RUN_ROOT'] = str(root)
        import run_paired as runner
        (root / 'code').mkdir()
        (root / 'data/chestagentbench').mkdir(parents=True)
        (root / 'code/protocol.json').write_text(json.dumps(plan))
        (root / 'runtime.private.json').write_text(json.dumps({'planner_path': 'unused'}))
        import time
        (root / 'guard.json').write_text(json.dumps({'time': time.time(), 'stop': False}))
        rows = [{'full_question_id': x['full_question_id'], 'images': x['images'],
                 'question': f'Fixture question {i}', 'answer': 'F',
                 'explanation': 'GOLD_EXPLANATION_MUST_NOT_REACH_AGENT'} for i, x in enumerate(plan['panel'])]
        (root / 'data/chestagentbench/metadata.jsonl').write_text('\n'.join(json.dumps(x) for x in rows))
        observed = []
        class Model:
            def __init__(self, *args):
                self.calls = self.tokens = 0
            def bind_tools(self, tools):
                return self
            def start_episode(self):
                self.calls = self.tokens = 0
        class Agent:
            def __init__(self, model, tools, **kwargs):
                self.workflow = self
                self.model = model.model
            def invoke(self, value, config):
                content = value['messages'][0].content
                text = content[-1]['text']
                assert 'GOLD_EXPLANATION' not in text and '"answer"' not in text
                assert content[0]['type'] == 'image'
                observed.append(content)
                self.model.calls = 1
                if 'Fixture question 3\n' in text and plan['arms']['B_consult'] in text:
                    raise RuntimeError('injected episode failure')
                return {'messages': [SimpleNamespace(content='Answer: F')]}
        runner.LocalVisionModel = Model
        runner.load_native = lambda *args: ([], {}, Agent, 'identical native system')
        assert runner.main() == 2
        summary = json.loads((root / 'FINISHED.json').read_text())
        assert summary['denominator'] == 24 and len(observed) == 24
        assert summary['arms']['A_native']['correct'] == 12
        assert summary['arms']['B_consult']['correct'] == 11
        assert summary['arms']['B_consult']['runtime_error_episodes'] == 1
        for index in range(0, 24, 2):
            left, right = observed[index:index + 2]
            assert left[:-1] == right[:-1]
            normalize = lambda text: text.replace('\n\n' + plan['arms']['B_consult'], '')
            assert normalize(left[-1]['text']) == normalize(right[-1]['text'])
        print('PAIRED_DESIGN_OK: 24 denominator, one injected failure retained, only instruction differs; model_generations=0')


if __name__ == '__main__':
    main()
