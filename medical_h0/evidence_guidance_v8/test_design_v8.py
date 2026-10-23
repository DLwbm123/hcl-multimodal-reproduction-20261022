"""CPU-only checks of policy isolation, source scopes and reference paths."""
import copy
import unittest
import bootstrap_v8
from fixtures_v8 import schedule,evidence_cases
from fixtures_v7 import expected
from adapter_v6 import SYSTEM
from adapter_v8 import policy_system,GUIDANCE
from adapter_v7 import collect_raw_input
from component_state import ComponentState
from tools_v5 import RestrictedTools
from audit_rules import reference_bindings
from evaluate_v7 import evaluate


class Design(unittest.TestCase):
    def test_prompt_and_pair_isolation(self):
        plan=schedule();self.assertEqual(len(plan),24)
        self.assertEqual(policy_system(SYSTEM,'R'),SYSTEM)
        self.assertEqual(policy_system(SYSTEM,'I'),SYSTEM+'\n\n'+GUIDANCE)
        self.assertNotIn('fact_',GUIDANCE)
        self.assertEqual([expected(c)['outcome'] for c in evidence_cases()],
                         ['answerable','answerable','answerable','conflict','conflict','missing'])
        for cid in {c['case_id'] for c in plan}:
            pair=[c for c in plan if c['case_id']==cid]
            self.assertEqual({c['arm'] for c in pair},{'R','I'})
            for k in ('prompt','rows','target'):self.assertEqual(pair[0][k],pair[1][k])

    def test_all_reference_paths(self):
        for c in [c for c in schedule() if c['arm']=='R']:
            s=ComponentState(RestrictedTools(c['rows'],c['target']),c['stage'])
            collect_raw_input(s,'B')
            self.assertEqual(len(s.read_row_ids),len(c['rows']))
            if c['stage']=='navigation':
                truth=expected(c)
                if truth['outcome']=='answerable':_,error=s.call('report_evidence',{'fact_ids':truth['fact_ids']})
                else:_,error=s.call('abstain',{'reason':truth['outcome']+' evidence'})
            else:
                s.call('get_calculator',{'calculator_id':c['target']['calculator_id']})
                refs=reference_bindings(c['target']['calculator_id'],c['rows'],c['target']['patient_id'],c['target']['episode_id'])
                args={k:v for k,v in c['target'].items() if k!='output_type'}
                args['bindings']=[{'parameter':k,'value':v['value'],'fact_ids':[g[0] for g in v['required_source_groups']],
                                   'transformation':v['derivation']} for k,v in refs.items()]
                result,error=s.call('execute_calculator',args);self.assertFalse(error)
                _,error=s.call('submit_result',{'result_id':result['result_id']})
            self.assertFalse(error)
            record={'answer':s.tools.answer,'execution_state':s.snapshot(),'result_ledger':s.results,'trace':[]}
            self.assertTrue(evaluate(record,c)['preliminary_correct'])

    def test_omitted_operand_stays_untrusted(self):
        c=next(c for c in schedule() if c['id']=='P01_R')
        s=ComponentState(RestrictedTools(c['rows'],c['target']),'mapping');collect_raw_input(s,'B')
        args={k:v for k,v in c['target'].items() if k!='output_type'}
        args['bindings']=[{'parameter':'opioid_name','value':'oxycodone','fact_ids':['fact_0003'],'transformation':'lowercase drug name'},
                          {'parameter':'daily_dose_mg','value':10,'fact_ids':['fact_0005'],'transformation':'2.5 times 4'}]
        out,error=s.call('execute_calculator',args);self.assertFalse(error)
        s.call('submit_result',{'result_id':out['result_id']})
        score=evaluate({'answer':s.tools.answer,'execution_state':s.snapshot(),'result_ledger':s.results,'trace':[]},c)
        self.assertTrue(score['numeric_correct']);self.assertFalse(score['preliminary_correct'])
        # The intervention must not weaken the evaluator to accept a right number.
        self.assertIn('daily_dose_mg:unsupported_or_incomplete_sources',score['result_audits']['result_001']['reasons'])


if __name__=='__main__':unittest.main()
