"""Post-run evaluator. The solver never imports this module."""
import math
from audit_data import canonical_output
from audit_rules import reference_bindings, Unresolved


def equal_value(actual, expected, derived=False):
    if type(expected) is bool:
        return type(actual) is bool and actual == expected
    if isinstance(expected, str):
        return type(actual) is str and actual == expected
    if type(actual) not in (int, float) or not math.isfinite(actual):
        return False
    return math.isclose(actual, expected, rel_tol=5e-4 if derived else 1e-8, abs_tol=1e-6)


def audit_result(result, target):
    if result['calculator_id'] != target['calculator_id'] or result['patient_id'] != target['patient_id'] or result['episode_id'] != target['episode_id']:
        return {'status': 'incorrect', 'reasons': ['identity_or_scope']}
    try:
        expected = reference_bindings(target['calculator_id'], target['rows'], target['patient_id'], target['episode_id'])
    except (Unresolved, ValueError, KeyError) as exc:
        return {'status': 'unverified', 'reasons': ['unresolved_evidence_rule'], 'detail': str(exc)}
    bindings = result.get('bindings', {})
    if set(bindings) != set(expected) or set(result.get('parameters', {})) != set(expected):
        return {'status': 'incorrect', 'reasons': ['parameter_set']}
    reasons = []
    for key, ref in expected.items():
        b = bindings[key]
        derived = ref['derivation'] not in ('direct', 'category', 'ais_category', 'roman_class', 'same_patient_stable_attribute')
        if not equal_value(b['value'], ref['value'], derived):
            reasons.append(key + ':parameter_value')
        if b['value'] != result['parameters'][key]:
            reasons.append(key + ':ledger_binding_mismatch')
        cited = set(b['fact_ids'])
        groups = [set(g) for g in ref['required_source_groups']]
        relevant = set().union(*groups)
        if not cited <= relevant or any(not cited & group for group in groups):
            reasons.append(key + ':unsupported_or_incomplete_sources')
    # Full formula value must also agree; parameter rounding tolerance cannot
    # excuse crossing a discrete score boundary.
    if not equal_value(result['value'], target['reference_output']['value'], derived=True):
        reasons.append('formula_output')
    return {'status': 'incorrect' if reasons else 'supported', 'reasons': reasons,
            'basis': 'frozen visible-fact rules and parameter-specific admissible evidence; transformation prose retained for audit, not used as self-certification'}


def score(record, target):
    ledger = record.get('result_ledger', [])
    audit = {}
    for result in ledger:
        rid = result['result_id']
        if result['output_type'] == 'raw':
            audit[rid] = audit_result(result, target)
        else:
            parent = next((r for r in ledger if r['result_id'] == result.get('parent_result_id')), None)
            if not parent or parent['result_id'] not in audit:
                audit[rid] = {'status': 'incorrect', 'reasons': ['missing_parent']}
            elif result.get('conversion') != 'spesi_risk_v1' or parent['calculator_id'] != 'spesi' or result['value'] != ('low' if parent['value'] == 0 else 'high'):
                audit[rid] = {'status': 'incorrect', 'reasons': ['invalid_derivation']}
            else:
                audit[rid] = dict(audit[parent['result_id']])
    answer = record.get('answer')
    numeric = trusted = False
    status = answer.get('status') if answer else 'unfinished'
    if status == 'success':
        result = next((r for r in ledger if r['result_id'] == answer.get('result_id')), None)
        if result:
            expected = target['canonical_benchmark']
            try:
                actual = canonical_output(answer['value'], target['calculator_id'])
                numeric = actual == expected if isinstance(expected, str) else abs(actual-expected) <= max(.1, abs(expected)*.02)
            except (ValueError, TypeError):
                numeric = False
            linked = all(answer.get(k) == result.get(k) for k in ('calculator_id', 'patient_id', 'episode_id', 'value'))
            typed = result['output_type'] == ('risk_category' if target['calculator_id'] == 'spesi' else 'raw')
            trusted = numeric and linked and typed and audit[result['result_id']]['status'] == 'supported'
    return {'numeric_correct': numeric, 'trusted_correct': trusted, 'state_report': status,
            'accepted_computations': len(ledger), 'result_audits': audit,
            'unverified_results': sum(r['status'] == 'unverified' for r in audit.values()),
            'semantic_error_results': sum(r['status'] == 'incorrect' for r in audit.values())}
