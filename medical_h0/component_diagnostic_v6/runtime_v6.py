"""Explicit terminal tools and durable model-visible evidence state."""
import copy
import json
import bootstrap
from jsonschema import Draft202012Validator, ValidationError
from runtime_v5 import ExecutionState as PreviousState
from tools_v5 import SCHEMAS as PREVIOUS_SCHEMAS, schema

SCHEMAS = [copy.deepcopy(s) for s in PREVIOUS_SCHEMAS if s['function']['name'] != 'submit_answer'] + [
    schema('submit_result', 'Explicitly submit an existing immutable result. This never changes its value.',
           {'result_id': {'type': 'string'}}, ['result_id']),
    schema('abstain', 'End this task without an answer when evidence is insufficient or execution cannot be completed.',
           {'reason': {'type': 'string', 'minLength': 1, 'maxLength': 500, 'pattern': r'\S'}}, ['reason']),
]
SCHEMAS[0]['function']['description'] += ' Omit offset to read the next unread page. Already-read facts and observation IDs remain in the persistent evidence state.'


class ExecutionState(PreviousState):
    def __init__(self, tools):
        super().__init__(tools)
        self.schemas = copy.deepcopy(SCHEMAS)
        self.validators = {s['function']['name']: Draft202012Validator(s['function']['parameters']) for s in self.schemas}
        self.pages = []
        self.observation_tools = {}

    def prepare(self, turn):
        self.turn = turn
        return [s for s in self.schemas if s['function']['name'] in self.allowed()]

    def allowed(self):
        if self.phase == 'done':
            return set()
        return set(self.validators)

    def next_offset(self):
        return next((i for i in range(len(self.tools.rows)) if f'fact_{i+1:04d}' not in self.read_row_ids), len(self.tools.rows))

    def dispatch(self, name, args):
        if name == 'submit_result':
            return super().dispatch('submit_answer', {'status': 'success', 'result_id': args['result_id']})
        if name == 'abstain':
            return super().dispatch('submit_answer', {'status': 'abstain', 'reason': args['reason']})
        result = super().dispatch(name, args)
        if name == 'read_facts':
            self.pages.append({'offset': args['offset'], 'count': len(result['rows']), 'next_offset': result['next_offset']})
        return result

    def call(self, name, args):
        self.request_count += 1
        if self.request_count > 80:
            return {'error': 'tool_call_limit'}, True
        try:
            if name not in self.allowed():
                raise ValueError('Tool unavailable in current phase')
            self.validators[name].validate(args)
            args = copy.deepcopy(args)
            if name == 'read_facts':
                args.setdefault('offset', self.next_offset())
                args.setdefault('limit', 100)
            key = name + ':' + json.dumps(args, sort_keys=True, allow_nan=False)
            if name == 'get_observation':
                if args['observation_id'] not in self.observations:
                    raise ValueError('Unknown observation_id')
                saved, error = self.observations[args['observation_id']]
                return {**copy.deepcopy(saved), 'retrieved': True}, error
            if name not in ('submit_result', 'abstain') and key in self.cache:
                oid, _ = self.cache[key]
                self.duplicates += 1
                self.no_progress += 1
                return {**copy.deepcopy(self.observations[oid][0]), 'cached': True}, False
            out, error = self.dispatch(name, args), False
            self.no_progress = 0
        except (ValueError, TypeError, KeyError, ValidationError, ArithmeticError) as exc:
            detail = exc.message if isinstance(exc, ValidationError) else str(exc)
            out, error = {'error': type(exc).__name__, 'detail': detail[:800]}, True
            self.errors.append({'tool': name, **out})
            self.no_progress += 1
        oid = f'observation_{len(self.observations)+1:03d}'
        out = {**out, 'observation_id': oid}
        self.observations[oid] = (copy.deepcopy(out), error)
        self.observation_tools[oid] = name
        # Failed calls depend on mutable read state and must be re-evaluated.
        if not error:
            self.cache[key] = (oid, False)
        return out, error

    def snapshot(self):
        state = super().snapshot()
        state['evidence_state'] = {
            'total_facts': len(self.tools.rows),
            'next_unread_offset': self.next_offset() if self.next_offset() < len(self.tools.rows) else None,
            'pages': copy.deepcopy(self.pages),
            'observations': [{'observation_id': oid, 'tool': self.observation_tools[oid],
                              'status': 'error' if saved[1] else 'success'} for oid, saved in self.observations.items()],
            'read_facts': [{**copy.deepcopy(self.fact_rows[fid]), 'fact_id': fid} for fid in sorted(self.read_row_ids)],
        }
        return state
