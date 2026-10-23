"""Restrict each diagnostic to one capability; no benchmark task access."""
import copy
import bootstrap
from jsonschema import Draft202012Validator
from tools_v5 import schema
from runtime_v6 import ExecutionState


class ComponentState(ExecutionState):
    def __init__(self, tools, stage):
        super().__init__(tools)
        self.stage = stage
        if stage == 'submission':
            names = {'submit_result', 'abstain'}
            self.schemas = [s for s in self.schemas if s['function']['name'] in names]
        elif stage == 'navigation':
            names = {'read_facts', 'get_observation', 'abstain'}
            self.schemas = [s for s in self.schemas if s['function']['name'] in names]
            read = self.schemas[0]['function']['parameters']['properties']['limit']
            read['maximum'] = 4
            self.schemas.append(schema('report_evidence', 'Explicitly report supporting fact IDs after reading them. This diagnostic performs no clinical calculation.',
                {'fact_ids': {'type': 'array', 'minItems': 1, 'uniqueItems': True, 'items': {'type': 'string'}}}, ['fact_ids']))
        self.validators = {s['function']['name']: Draft202012Validator(s['function']['parameters']) for s in self.schemas}

    def call(self, name, args):
        if self.stage == 'navigation' and name == 'read_facts' and isinstance(args, dict):
            args = copy.deepcopy(args); args.setdefault('limit', 4)
        return super().call(name, args)

    def dispatch(self, name, args):
        if name == 'report_evidence':
            for fid in args['fact_ids']:
                if fid not in self.read_row_ids:
                    raise ValueError('Source fact must exist and have been read')
                if any(self.fact_rows[fid][k] != self.target[k] for k in ('patient_id', 'episode_id')):
                    raise ValueError('Evidence outside the target scope')
            self.tools.answer = {'status': 'success', 'fact_ids': list(args['fact_ids'])}
            self.phase = 'done'
            return {'reported': True}
        return super().dispatch(name, args)
