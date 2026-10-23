"""Synthetic regression checks for the diagnosed interface/state defects."""
import copy
import unittest
import bootstrap
from tools_v5 import RestrictedTools
from runtime_v5 import ExecutionState as OldState
from runtime_v6 import ExecutionState
from component_state import ComponentState
from fixtures import row, target, result, cases
from audit_rules import reference_bindings
from catalog_v5 import execute
from score_v5 import score
from audit_data import canonical_output


def binding():
    return {'calculator_id': 'rass', 'patient_id': 'synthetic_patient', 'episode_id': 'synthetic_visit',
            'bindings': [{'parameter': 'rass_score', 'value': 1, 'fact_ids': ['fact_0002'], 'transformation': 'Map the explicit Restless category to +1.'}]}


class Components(unittest.TestCase):
    def state(self, cls=ExecutionState):
        return cls(RestrictedTools([row('distractor', 0), row('patient_description', 'Restless (+1)')], target('rass')))

    def test_state_dependent_failure_recovers(self):
        for cls, should_fail in [(OldState, True), (ExecutionState, False)]:
            s = self.state(cls); s.call('read_facts', {'offset': 0, 'limit': 1})
            self.assertTrue(s.call('execute_calculator', binding())[1])
            s.call('read_facts', {'offset': 1, 'limit': 1})
            self.assertEqual(s.call('execute_calculator', binding())[1], should_fail)

    def test_explicit_terminal_contract(self):
        s = self.state(); s.results = [result('result_001', 1, cid='rass')]
        self.assertNotIn('submit_answer', s.validators)
        self.assertTrue(s.call('submit_result', {'result_id': 'result_001', 'reason': 'extra'})[1])
        self.assertTrue(s.call('submit_result', {'result_id': 'missing'})[1])
        self.assertFalse(s.call('submit_result', {'result_id': 'result_001'})[1])
        self.assertEqual(s.tools.answer['value'], 1)
        s = self.state()
        self.assertTrue(s.call('abstain', {'reason': ' '})[1])
        self.assertFalse(s.call('abstain', {'reason': 'Required evidence absent'})[1])

    def test_success_cache_preserves_body_and_immutable_result(self):
        s = self.state(); a={'offset': 0, 'limit': 2}
        first, _ = s.call('read_facts', a); again, error = s.call('read_facts', a)
        self.assertFalse(error); self.assertEqual(first['rows'], again['rows'])
        r, _ = s.call('execute_calculator', binding()); repeated, _ = s.call('execute_calculator', binding())
        self.assertEqual(r['result_id'], repeated['result_id']); self.assertEqual(len(s.results), 1)

    def test_persistent_facts_and_next_page(self):
        s = self.state(); s.call('read_facts', {'limit': 1})
        self.assertEqual(s.snapshot()['evidence_state']['next_unread_offset'], 1)
        for _ in range(4):s.call('get_calculator', {'calculator_id': 'rass'})
        self.assertEqual(s.snapshot()['evidence_state']['read_facts'][0]['fact_id'], 'fact_0001')
        second, _ = s.call('read_facts', {'limit': 1})
        self.assertEqual(second['rows'][0]['fact_id'], 'fact_0002')
        self.assertIsNone(s.snapshot()['evidence_state']['next_unread_offset'])
        self.assertTrue(s.snapshot()['evidence_state']['observations'])

    def test_scope_and_output_type_remain_enforced(self):
        s=self.state();s.tools.rows[1]['episode_id']='old';s.fact_rows['fact_0002']['episode_id']='old'
        s.call('read_facts', {}); self.assertTrue(s.call('execute_calculator', binding())[1])
        s=self.state();s.target['output_type']='risk_category';s.results=[result('result_001',1,cid='rass')]
        self.assertTrue(s.call('submit_result', {'result_id':'result_001'})[1])

    def test_synthetic_navigation_and_mapping_reference_paths(self):
        for c in cases():
            if c['stage']=='submission':continue
            s=ComponentState(RestrictedTools(c['rows'],c['target']),c['stage'])
            if c['stage']=='navigation':
                while s.next_offset()<len(c['rows']):self.assertFalse(s.call('read_facts',{})[1])
                self.assertFalse(s.call('report_evidence',{'fact_ids':c['expected_fact_ids']})[1])
            else:
                s.call('read_facts',{})
                refs=reference_bindings(c['target']['calculator_id'],c['rows'],c['target']['patient_id'],c['target']['episode_id'])
                params={k:v['value'] for k,v in refs.items()};out=execute(c['target']['calculator_id'],params)
                args={k:v for k,v in c['target'].items() if k!='output_type'}
                args['bindings']=[{'parameter':k,'value':v['value'],'fact_ids':[g[0] for g in v['required_source_groups']],'transformation':v['derivation']} for k,v in refs.items()]
                r,error=s.call('execute_calculator',args);self.assertFalse(error)
                self.assertFalse(s.call('submit_result',{'result_id':r['result_id']})[1])
                t={**c['target'],'rows':c['rows'],'reference_output':out,'canonical_benchmark':canonical_output(out['value'],c['target']['calculator_id'])}
                self.assertTrue(score({'answer':s.tools.answer,'result_ledger':s.results},t)['trusted_correct'])


if __name__=='__main__':unittest.main()
