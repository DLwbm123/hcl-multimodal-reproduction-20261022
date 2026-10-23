"""Structural validation and immutable results; no evaluator access."""
import copy
import json
import unicodedata
from jsonschema import Draft202012Validator, ValidationError
from tools_v5 import SCHEMAS
from catalog_v5 import details, execute


class ExecutionState:
    def __init__(self, tools):
        self.tools = tools
        self.target = tools.target
        self.results = []
        self.observations = {}
        self.cache = {}
        self.read_row_ids = set()
        self.fact_rows = {f'fact_{i+1:04d}': r for i, r in enumerate(tools.rows)}
        self.errors = []
        self.phase = 'facts'
        self.turn = self.request_count = self.duplicates = self.no_progress = 0
        self.validators = {s['function']['name']: Draft202012Validator(s['function']['parameters']) for s in SCHEMAS}

    def prepare(self, turn):
        self.turn = turn
        return [s for s in SCHEMAS if s['function']['name'] in self.allowed()]

    def allowed(self):
        if self.phase == 'done':
            return set()
        if self.phase == 'facts':
            return {'read_facts', 'submit_answer'}
        return set(self.validators)

    def result(self, rid):
        found = [r for r in self.results if r['result_id'] == rid]
        if len(found) != 1:
            raise ValueError('Unknown result_id')
        return found[0]

    def execute(self, args):
        for k in ('calculator_id', 'patient_id', 'episode_id'):
            if args[k] != self.target[k]:
                raise ValueError('Invocation conflicts with explicit target ' + k)
        params, bindings = {}, {}
        for b in args['bindings']:
            key = b['parameter']
            if key in params:
                raise ValueError('Duplicate parameter binding: ' + key)
            for fid in b['fact_ids']:
                if fid not in self.read_row_ids:
                    raise ValueError('Source fact must exist and have been read')
                f = self.fact_rows[fid]
                if f['patient_id'] != self.target['patient_id']:
                    raise ValueError('Cross-patient reference forbidden')
                if f['episode_id'] != self.target['episode_id']:
                    stable_sex = (args['calculator_id'] == 'corrected_qt' and key == 'sex'
                                  and unicodedata.normalize('NFKC', f['field_name']).casefold().strip() == 'sex')
                    if not stable_sex:
                        raise ValueError('This parameter requires target-episode evidence')
            params[key] = b['value']
            bindings[key] = copy.deepcopy(b)
        out = execute(args['calculator_id'], params)
        bound = {**out, 'result_id': f'result_{len(self.results)+1:03d}',
                 'patient_id': args['patient_id'], 'episode_id': args['episode_id'],
                 'parameters': copy.deepcopy(params), 'bindings': bindings,
                 'output_type': 'raw', 'semantic_certification': 'not certified by execution'}
        self.results.append(bound)
        return {k: v for k, v in bound.items() if k not in ('bindings', 'parameters')}

    def dispatch(self, name, args):
        if name == 'read_facts':
            offset, limit = args.get('offset', 0), args.get('limit', 100)
            rows = [{**r, 'fact_id': f'fact_{offset+i+1:04d}'} for i, r in enumerate(self.tools.rows[offset:offset+limit])]
            self.read_row_ids.update(r['fact_id'] for r in rows)
            self.phase = 'work'
            return {'rows': rows, 'total': len(self.tools.rows), 'next_offset': offset+limit if offset+limit < len(self.tools.rows) else None}
        if name == 'get_calculator':
            if args['calculator_id'] != self.target['calculator_id']:
                raise ValueError('Only the explicitly requested calculator is in scope')
            return details(args['calculator_id'])
        if name == 'execute_calculator':
            return self.execute(args)
        if name == 'derive_output':
            parent = self.result(args['result_id'])
            if parent['calculator_id'] != 'spesi' or parent['output_type'] != 'raw':
                raise ValueError('spesi_risk_v1 requires a raw sPESI score')
            value = parent['value']
            if type(value) not in (int, float) or value < 0:
                raise ValueError('Invalid parent score')
            result = {k: copy.deepcopy(parent[k]) for k in ('calculator_id', 'name', 'patient_id', 'episode_id', 'formula_version')}
            result.update(result_id=f'result_{len(self.results)+1:03d}', parent_result_id=parent['result_id'],
                          conversion='spesi_risk_v1', value='low' if value == 0 else 'high',
                          unit='risk category', output_type='risk_category')
            self.results.append(result)
            return copy.deepcopy(result)
        if name == 'submit_answer':
            if args['status'] == 'success':
                if set(args) != {'status', 'result_id'}:
                    raise ValueError('Success must contain only status and result_id')
                r = self.result(args['result_id'])
                if r['output_type'] != self.target['output_type']:
                    raise ValueError('Result output type differs from requested output contract')
                self.tools.answer = {'status': 'success', 'result_id': r['result_id'],
                                     'calculator_id': r['calculator_id'], 'patient_id': r['patient_id'],
                                     'episode_id': r['episode_id'], 'value': r['value'], 'unit': r['unit']}
            else:
                if set(args) != {'status', 'reason'} or not args['reason'].strip():
                    raise ValueError('Abstention/failure needs a reason and cannot include a result')
                self.tools.answer = dict(args)
            self.phase = 'done'
            return {'reported': True, 'status': self.tools.answer['status']}
        raise ValueError('Unknown tool')

    def call(self, name, args):
        self.request_count += 1
        if self.request_count > 80:
            return {'error': 'tool_call_limit'}, True
        try:
            if name not in self.allowed():
                raise ValueError('Tool unavailable in current phase')
            self.validators[name].validate(args)
            key = name + ':' + json.dumps(args, sort_keys=True, allow_nan=False)
            if name == 'get_observation':
                oid = args['observation_id']
                if oid not in self.observations:
                    raise ValueError('Unknown observation_id')
                saved, error = self.observations[oid]
                return {**copy.deepcopy(saved), 'retrieved': True}, error
            if name != 'submit_answer' and key in self.cache:
                oid, error = self.cache[key]
                self.duplicates += 1
                self.no_progress += 1
                return {'observation_id': oid, 'cached': True, 'status': 'error' if error else 'success'}, error
            out = self.dispatch(name, args)
            error = False
            self.no_progress = 0
        except (ValueError, TypeError, KeyError, ValidationError, ArithmeticError) as exc:
            detail = exc.message if isinstance(exc, ValidationError) else str(exc)
            out, error = {'error': type(exc).__name__, 'detail': detail[:800]}, True
            self.errors.append({'tool': name, **out})
            self.no_progress += 1
        oid = f'observation_{len(self.observations)+1:03d}'
        out = {**out, 'observation_id': oid}
        self.observations[oid] = (copy.deepcopy(out), error)
        if 'key' in locals():
            self.cache[key] = (oid, error)
        return out, error

    def snapshot(self):
        return {'phase': self.phase, 'remaining_turns': 24-self.turn,
                'target': self.target, 'read_fact_count': len(self.read_row_ids),
                'computed_results': [{k: r[k] for k in ('result_id', 'calculator_id', 'episode_id', 'value', 'output_type')} for r in self.results],
                'recent_errors': self.errors[-4:], 'tool_requests': self.request_count,
                'submitted': self.tools.answer is not None, 'no_progress_events': self.no_progress,
                'note': 'Result values are immutable. Existence of a result does not certify source semantics.'}
