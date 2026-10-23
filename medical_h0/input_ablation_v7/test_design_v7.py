"""CPU checks for paired isolation, exact scoring and all reference paths."""
import copy
import unittest
import bootstrap_v7
from fixtures_v7 import navigation_cases, expected, schedule
from adapter_v7 import collect_raw_input
from component_state import ComponentState
from tools_v5 import RestrictedTools
from evaluate_v7 import evaluate
from audit_rules import reference_bindings
from catalog_v5 import execute


def state(case):
    return ComponentState(RestrictedTools(copy.deepcopy(case['rows']),copy.deepcopy(case['target'])),case['stage'])


def record(s):
    return {'answer':s.tools.answer,'execution_state':s.snapshot(),'trace':[], 'result_ledger':s.results}


class Design(unittest.TestCase):
    def test_pairing_and_new_panel(self):
        plan = schedule()
        self.assertEqual(len(plan),30)
        self.assertEqual(len({c['id'] for c in plan}),30)
        self.assertEqual([expected(c)['outcome'] for c in navigation_cases()].count('answerable'),8)
        self.assertEqual([expected(c)['outcome'] for c in navigation_cases()].count('missing'),2)
        self.assertEqual([expected(c)['outcome'] for c in navigation_cases()].count('conflict'),2)
        for c in navigation_cases():
            pair = [p for p in plan if p['case_id']==c['id']]
            self.assertEqual({p['arm'] for p in pair},{'A','B'})
            for k in ('prompt','rows','target'):
                self.assertEqual(pair[0][k],pair[1][k])
            self.assertEqual(len(c['rows']),12)
            self.assertNotIn('expected',c['prompt'])
        self.assertEqual(sum(c['stage']=='mapping' for c in plan),6)

    def test_only_raw_input_acquisition_differs(self):
        for c in navigation_cases():
            a,b = state(c),state(c)
            self.assertEqual(collect_raw_input(a,'A'),[])
            receipts = collect_raw_input(b,'B')
            self.assertEqual(len(receipts),3)
            self.assertEqual(len(a.read_row_ids),0)
            self.assertEqual(len(b.read_row_ids),12)
            self.assertEqual(a.schemas,b.schemas)
            self.assertEqual(a.results,b.results)
            self.assertIsNone(b.tools.answer)
            self.assertEqual([r['field_name'] for r in b.snapshot()['evidence_state']['read_facts']],
                             [r['field_name'] for r in c['rows']])

    def test_navigation_reference_paths_and_scope_rejection(self):
        for c in navigation_cases():
            s = state(c); collect_raw_input(s,'B'); truth = expected(c)
            if truth['outcome']=='answerable':
                _,error = s.call('report_evidence',{'fact_ids':truth['fact_ids']})
            else:
                _,error = s.call('abstain',{'reason':truth['outcome']+' evidence'})
            self.assertFalse(error)
            self.assertTrue(evaluate(record(s),c)['preliminary_correct'])
        c=navigation_cases()[5];s=state(c);collect_raw_input(s,'B')
        self.assertTrue(s.call('report_evidence',{'fact_ids':['fact_0002']})[1])

    def test_false_termination_not_promoted(self):
        c=navigation_cases()[2];s=state(c);s.call('read_facts',{})
        s.call('report_evidence',{'fact_ids':['fact_0001']})
        result=evaluate(record(s),c)
        self.assertFalse(result['preliminary_correct'])
        self.assertFalse(result['required_evidence_visible_at_end'])
        self.assertTrue(result['incorrect_termination_with_unread_facts'])
        for c in navigation_cases()[8:]:
            s=state(c);s.call('read_facts',{});s.call('abstain',{'reason':'Missing evidence'})
            self.assertFalse(evaluate(record(s),c)['preliminary_correct'])
        c=navigation_cases()[0];s=state(c);s.call('read_facts',{})
        s.call('report_evidence',{'fact_ids':expected(c)['fact_ids']})
        result=evaluate(record(s),c)
        self.assertTrue(result['ended_with_unread_facts'])
        self.assertFalse(result['incorrect_termination_with_unread_facts'])

    def test_mapping_reference_paths_without_navigation_gate(self):
        for c in [c for c in schedule() if c['stage']=='mapping']:
            s=state(c);collect_raw_input(s,'B')
            s.call('get_calculator',{'calculator_id':c['target']['calculator_id']})
            refs=reference_bindings(c['target']['calculator_id'],c['rows'],c['target']['patient_id'],c['target']['episode_id'])
            args={k:v for k,v in c['target'].items() if k!='output_type'}
            args['bindings']=[{'parameter':k,'value':v['value'],'fact_ids':[g[0] for g in v['required_source_groups']],
                'transformation':v['derivation']} for k,v in refs.items()]
            result,error=s.call('execute_calculator',args);self.assertFalse(error)
            self.assertFalse(s.call('submit_result',{'result_id':result['result_id']})[1])
            self.assertTrue(evaluate(record(s),c)['preliminary_correct'])


if __name__=='__main__':unittest.main()
