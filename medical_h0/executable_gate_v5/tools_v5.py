"""One binding record per parameter; no direct numeric submission path."""
import copy


def schema(name, description, properties, required):
    return {'type': 'function', 'function': {'name': name, 'description': description,
        'parameters': {'type': 'object', 'properties': properties, 'required': required, 'additionalProperties': False}}}


BINDING = {'type': 'object', 'properties': {
    'parameter': {'type': 'string'}, 'value': {'type': ['number', 'string', 'boolean']},
    'fact_ids': {'type': 'array', 'minItems': 1, 'uniqueItems': True, 'items': {'type': 'string'}},
    'transformation': {'type': 'string', 'minLength': 1, 'maxLength': 500}},
    'required': ['parameter', 'value', 'fact_ids', 'transformation'], 'additionalProperties': False}

SCHEMAS = [
    schema('read_facts', 'Read paginated facts. All task records and distractors remain visible. Generated timestamp, unit and specimen hints are not authoritative clinical metadata; use explicit field/value text and the declared source policy.',
           {'offset': {'type': 'integer', 'minimum': 0}, 'limit': {'type': 'integer', 'minimum': 1, 'maximum': 100}}, []),
    schema('get_calculator', 'Read the required formula profile, parameters, units, categories and source policy before executing.',
           {'calculator_id': {'type': 'string'}}, ['calculator_id']),
    schema('execute_calculator', 'Execute the requested formula. Give each parameter once as a binding with your value, supporting fact IDs and extraction/conversion explanation. Do not guess absent measurements or treat unknown as false. Values and evidence are never corrected for you.',
           {'calculator_id': {'type': 'string'}, 'patient_id': {'type': 'string'}, 'episode_id': {'type': 'string'},
            'bindings': {'type': 'array', 'minItems': 1, 'maxItems': 32, 'items': BINDING}},
           ['calculator_id', 'patient_id', 'episode_id', 'bindings']),
    schema('derive_output', 'Create an immutable requested risk-category result from a successful raw score. The supported conversion is spesi_risk_v1: score zero -> low, score above zero -> high. This does not change the parent score.',
           {'result_id': {'type': 'string'}, 'conversion': {'enum': ['spesi_risk_v1']}}, ['result_id', 'conversion']),
    schema('get_observation', 'Recover a prior full observation by its observation_id.',
           {'observation_id': {'type': 'string'}}, ['observation_id']),
    schema('submit_answer', 'Explicitly report success by immutable result_id, or abstain/fail with a reason. Do not provide a replacement numeric value. A raw score cannot replace a requested risk category.',
           {'status': {'enum': ['success', 'abstain', 'failed']}, 'result_id': {'type': 'string'},
            'reason': {'type': 'string', 'maxLength': 500}}, ['status']),
]


class RestrictedTools:
    def __init__(self, rows, target):
        self.rows = copy.deepcopy(rows)
        self.target = copy.deepcopy(target)
        self.answer = None
