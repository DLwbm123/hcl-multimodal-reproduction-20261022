"""Synthetic contract tests; no benchmark patient records or model calls."""
import copy
import json
import tempfile
import time
import unittest
from pathlib import Path

from audit_rules import reference_bindings
from audit_data import canonical_output
from catalog_v5 import execute
from tools_v5 import RestrictedTools
from runtime_v5 import ExecutionState
from score_v5 import score
from budget_v5 import Budget, BudgetStop, atomic


def row(field, value, episode='e1', patient='synthetic'):
    return dict(patient_id=patient, field_name=field, value_text=value,
                observed_at='synthetic', episode_id=episode, unit='', sample_type='synthetic', source='synthetic')


def target(rows, cid='rass'):
    b = reference_bindings(cid, rows, 'synthetic', 'e1')
    p = {k: v['value'] for k, v in b.items()}
    out = execute(cid, p)
    return dict(calculator_id=cid, patient_id='synthetic', episode_id='e1', rows=rows,
                output_type='raw', reference_output=out, canonical_benchmark=canonical_output(out['value'], cid))


class GateTests(unittest.TestCase):
    def setUp(self):
        self.rows = [row('patient_description', 'Restless (+1)'), row('Unrelated', '1')]
        self.target = target(self.rows)
        self.tools = RestrictedTools(self.rows, {k: self.target[k] for k in ('calculator_id', 'patient_id', 'episode_id', 'output_type')})
        self.state = ExecutionState(self.tools)
        self.state.call('read_facts', {})

    def args(self, ids=None, value=1):
        return dict(calculator_id='rass', patient_id='synthetic', episode_id='e1',
                    bindings=[dict(parameter='rass_score', value=value, fact_ids=ids or ['fact_0001'], transformation='Map the explicit category using the definition')])

    def result(self, args=None):
        out, error = self.state.call('execute_calculator', args or self.args())
        self.assertFalse(error, out)
        return out

    def record(self):
        return dict(answer=self.tools.answer, result_ledger=self.state.results)

    def test_valid_path(self):
        r = self.result()
        out, err = self.state.call('submit_answer', {'status': 'success', 'result_id': r['result_id']})
        self.assertFalse(err, out)
        self.assertTrue(score(self.record(), self.target)['trusted_correct'])

    def test_numeric_right_source_wrong(self):
        r = self.result(self.args(['fact_0002']))
        self.state.call('submit_answer', {'status': 'success', 'result_id': r['result_id']})
        scored = score(self.record(), self.target)
        self.assertTrue(scored['numeric_correct'])
        self.assertFalse(scored['trusted_correct'])

    def test_equivalent_source_accepted(self):
        rows = self.rows + [row('patient_description', 'Restless (+1)')]
        self.target = target(rows)
        self.tools = RestrictedTools(rows, self.tools.target)
        self.state = ExecutionState(self.tools)
        self.state.call('read_facts', {})
        r = self.result(self.args(['fact_0003']))
        self.state.call('submit_answer', {'status': 'success', 'result_id': r['result_id']})
        self.assertTrue(score(self.record(), self.target)['trusted_correct'])

    def test_invalid_domain_and_type(self):
        for value in [9, True, 'restless', None]:
            self.assertTrue(self.state.call('execute_calculator', self.args(value=value))[1])
        self.assertFalse(self.state.results)

    def test_no_value_override_or_direct_value(self):
        r = self.result()
        for args in [dict(status='success', result_id=r['result_id'], value=3), dict(status='success', value=1)]:
            self.assertTrue(self.state.call('submit_answer', args)[1])
        self.assertIsNone(self.tools.answer)
        self.assertEqual(self.state.results[0]['value'], 1)

    def test_cross_patient_and_episode_and_unread(self):
        for extra in [row('patient_description', 'Restless (+1)', patient='other'), row('patient_description', 'Restless (+1)', episode='old')]:
            tools = RestrictedTools(self.rows + [extra], self.tools.target)
            state = ExecutionState(tools)
            self.assertTrue(state.call('execute_calculator', self.args())[1])
            state.call('read_facts', {})
            self.assertTrue(state.call('execute_calculator', self.args(['fact_0003']))[1])

    def test_explicit_abstention(self):
        self.assertTrue(self.state.call('submit_answer', {'status': 'abstain'})[1])
        self.assertFalse(self.state.call('submit_answer', {'status': 'abstain', 'reason': 'Missing evidence'})[1])
        self.assertFalse(score(self.record(), self.target)['trusted_correct'])

    def test_missing_and_duplicate_parameter(self):
        args = self.args(); args['bindings'] *= 2
        self.assertTrue(self.state.call('execute_calculator', args)[1])
        args['bindings'] = []
        self.assertTrue(self.state.call('execute_calculator', args)[1])

    def test_stable_attribute_cross_episode(self):
        rows = [row('Measured QT interval', '400 ms'), row('Heart rate', '60 bpm'),
                row('Correction formula', 'Bazett'), row('Sex', 'Male', episode='old')]
        t = target(rows, 'corrected_qt'); tools = RestrictedTools(rows, {k: t[k] for k in ('calculator_id', 'patient_id', 'episode_id', 'output_type')})
        state = ExecutionState(tools); state.call('read_facts', {})
        refs = reference_bindings('corrected_qt', rows, 'synthetic', 'e1')
        args = {k: t[k] for k in ('calculator_id', 'patient_id', 'episode_id')}
        args['bindings'] = [dict(parameter=k, value=v['value'], fact_ids=[g[0] for g in v['required_source_groups']], transformation=v['derivation']) for k, v in refs.items()]
        r, err = state.call('execute_calculator', args)
        self.assertFalse(err, r)
        state.call('submit_answer', dict(status='success', result_id=r['result_id']))
        self.assertTrue(score(dict(answer=tools.answer, result_ledger=state.results), t)['trusted_correct'])

    def test_declared_output_profiles(self):
        self.assertEqual(execute('anion_gap', dict(sodium=140, chloride=110, bicarbonate=20))['value'], 10)
        self.assertEqual(execute('mme_calculator', dict(opioid_name='oxycodone', daily_dose_mg=10))['value'], 15)
        with self.assertRaises(Exception):
            execute('anion_gap', dict(sodium=140, chloride=110, bicarbonate=20, albumin=2))
        self.assertEqual(canonical_output('44 ml/min/1.73 m²', 'ckd_epi_2021'), 44)
        with self.assertRaises(ValueError):
            canonical_output('44 arbitrary commentary', 'ckd_epi_2021')

    def test_boolean_error_is_model_error(self):
        rows = [row('Age ≥50', 'No'), row('HR ≥100', 'No'), row('O2 sat on room air <95%', 'No'),
                row('Unilateral leg swelling', 'No'), row('Hemoptysis', 'No'), row('Recent surgery or trauma', 'No'),
                row('Prior PE or DVT', 'No'), row('Hormone use', 'No')]
        t = target(rows, 'perc_rule')
        tools = RestrictedTools(rows, {k: t[k] for k in ('calculator_id', 'patient_id', 'episode_id', 'output_type')})
        state = ExecutionState(tools); state.call('read_facts', {})
        refs = reference_bindings('perc_rule', rows, 'synthetic', 'e1')
        args = {k: t[k] for k in ('calculator_id', 'patient_id', 'episode_id')}
        args['bindings'] = [dict(parameter=k, value=v['value'], fact_ids=[g[0] for g in v['required_source_groups']], transformation='explicit yes/no') for k, v in refs.items()]
        args['bindings'][0]['value'] = True
        r, err = state.call('execute_calculator', args)
        self.assertFalse(err, r)
        state.call('submit_answer', dict(status='success', result_id=r['result_id']))
        self.assertFalse(score(dict(answer=tools.answer, result_ledger=state.results), t)['trusted_correct'])

    def test_derived_result_immutable_and_parent_audited(self):
        rows = [row('Age, years', '40'), row('History of cancer', 'No'), row('History of chronic cardiopulmonary disease', 'No'),
                row('Heart rate, bpm', '80'), row('Systolic BP, mmHg', '120'), row('O2 saturation', '98%')]
        t = target(rows, 'spesi'); t['output_type'] = 'risk_category'
        tools = RestrictedTools(rows, {k: t[k] for k in ('calculator_id', 'patient_id', 'episode_id', 'output_type')})
        state = ExecutionState(tools); state.call('read_facts', {})
        refs = reference_bindings('spesi', rows, 'synthetic', 'e1')
        args = {k: t[k] for k in ('calculator_id', 'patient_id', 'episode_id')}
        args['bindings'] = [dict(parameter=k, value=v['value'], fact_ids=[g[0] for g in v['required_source_groups']], transformation='explicit fact') for k, v in refs.items()]
        r, err = state.call('execute_calculator', args); self.assertFalse(err, r)
        self.assertTrue(state.call('submit_answer', dict(status='success', result_id=r['result_id']))[1])
        derived, err = state.call('derive_output', dict(result_id=r['result_id'], conversion='spesi_risk_v1'))
        self.assertFalse(err, derived)
        state.call('submit_answer', dict(status='success', result_id=derived['result_id']))
        self.assertEqual(state.results[0]['value'], 0)
        self.assertTrue(score(dict(answer=tools.answer, result_ledger=state.results), t)['trusted_correct'])

    def test_budget_attempt_and_token_reservation(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            atomic(p/'budget.json', dict(early_stop_epoch=time.time()+100, max_new_bytes=10000,
                    max_model_requests=2, max_input_tokens=100, max_output_tokens=20, max_output_per_generation=10))
            atomic(p/'guard_status.json', dict(time=time.time(), bytes_with_reserve=1))
            b = Budget(p); b.request('synthetic'); b.finish(status='context_rejected')
            self.assertEqual(b.state['requests'], 1); self.assertEqual(b.state['real_generations'], 0)
            b.request('synthetic'); b.reserve_tokens(50); b.generation_started(); b.finish(5)
            self.assertEqual(b.state['input_tokens'], 50); self.assertEqual(b.state['output_tokens'], 5)
            with self.assertRaises(BudgetStop): b.request('synthetic')
            with self.assertRaises(BudgetStop): b.check(time.time()+200)


if __name__ == '__main__':
    unittest.main()
