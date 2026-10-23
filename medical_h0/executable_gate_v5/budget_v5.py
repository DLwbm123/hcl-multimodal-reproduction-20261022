"""Per-attempt reservations. Counts failed requests separately from generations."""
import json
import time
from pathlib import Path


class BudgetStop(BaseException):
    pass


def atomic(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


class Budget:
    def __init__(self, root):
        self.root = Path(root)
        self.config = json.loads((self.root/'budget.json').read_text())
        self.state = {'requests': 0, 'real_generations': 0, 'input_tokens': 0, 'output_tokens': 0,
                      'reserved_input_tokens': 0, 'reserved_output_tokens': 0,
                      'active_request': None, 'updated_epoch': time.time()}
        if (self.root/'cost_state.json').exists():
            raise BudgetStop('Refuse budget restart')
        self.save()

    def save(self):
        self.state['updated_epoch'] = time.time()
        atomic(self.root/'cost_state.json', self.state)

    def check(self, now=None):
        now = time.time() if now is None else now
        if now >= self.config['early_stop_epoch'] or (self.root/'HARD_STOP.json').exists():
            raise BudgetStop('Wall-clock or independent guard stop')
        guard = json.loads((self.root/'guard_status.json').read_text())
        if now - guard['time'] > 30:
            raise BudgetStop('Guard status stale')
        if guard['bytes_with_reserve'] >= self.config['max_new_bytes']:
            raise BudgetStop('Storage budget reached')

    def request(self, target):
        self.check()
        if self.state['active_request'] is not None:
            raise BudgetStop('Prior reservation unresolved')
        if self.state['requests'] >= self.config['max_model_requests']:
            raise BudgetStop('Request cap reached')
        self.state['requests'] += 1
        self.state['active_request'] = {'number': self.state['requests'], 'target': target, 'generation_started': False}
        self.save()
        return self.state['requests']

    def reserve_tokens(self, count):
        maximum = self.config['max_output_per_generation']
        if self.state['input_tokens'] + count > self.config['max_input_tokens']:
            raise BudgetStop('Input token budget cannot cover next request')
        if self.state['output_tokens'] + maximum > self.config['max_output_tokens']:
            raise BudgetStop('Output token budget cannot reserve next generation')
        self.check()
        self.state['reserved_input_tokens'] = count
        self.state['reserved_output_tokens'] = maximum
        self.save()

    def generation_started(self):
        self.state['active_request']['generation_started'] = True
        self.state['real_generations'] += 1
        self.save()

    def finish(self, output_count=None, status='complete'):
        request = self.state['active_request']
        if request is None:
            return
        generated = request['generation_started']
        inp = self.state['reserved_input_tokens'] if generated else 0
        out = (self.state['reserved_output_tokens'] if output_count is None else output_count) if generated else 0
        self.state['input_tokens'] += inp
        self.state['output_tokens'] += out
        with (self.root/'request_ledger.jsonl').open('a') as f:
            f.write(json.dumps({**request, 'input_tokens': inp, 'output_tokens': out,
                               'output_count_exact': generated and output_count is not None,
                               'status': status, 'time': time.time()}) + '\n')
        self.state.update(reserved_input_tokens=0, reserved_output_tokens=0, active_request=None)
        self.save()
