"""Zero-generation native protocol and tool-boundary checks."""
import json
import os
from pathlib import Path


def main():
    from lagent import ReAct, ActionExecutor
    from agentlego.tools.calculator.python_calculator import Calculator
    from tool_worker import validate
    root = Path(os.environ['RUN_ROOT'])
    plan = json.loads((root / 'code/protocol.json').read_text())
    assert len(plan['panel']) == len({x['case_id'] for x in plan['panel']}) == 12
    assert not {x['case_id'] for x in plan['panel']} & set(plan['excluded_prior_cases'])
    class Fixture:
        calls = 0
        def chat(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return 'Thought: Calculate.\nAction: Calculator\nAction Input: {"expression":"2+3"}'
            assert '5' in messages[-1]['content']
            return 'Thought: Done.\nFinal Answer: 5'
    result = ReAct(Fixture(), ActionExecutor([Calculator().to_lagent()]), max_turn=3).chat('Compute 2+3')
    assert result.response.strip() == '5'
    assert result.actions[0].type == 'Calculator'
    assert len(result.actions) == 2
    path = str(root / 'canonical' / (plan['panel'][0]['full_question_id'] + '.png'))
    validate('OCR', {'image': path}, path)
    for name, args in [('OCR', {'image': '/etc/passwd'}),
                       ('Calculator', {'expression': '__import__("os")'}),
                       ('Calculator', {'expression': '(1).__class__'})]:
        try:
            validate(name, args, path)
        except ValueError:
            pass
        else:
            raise AssertionError('Unsafe tool input accepted')
    print('NATIVE_CPU_OK: unchanged ReAct executes real Calculator and consumes result; image/expression bounds; model_generations=0')


if __name__ == '__main__':
    main()
