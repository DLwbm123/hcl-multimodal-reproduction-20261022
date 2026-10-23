"""CPU reference paths and adversarial guard checks; zero model calls."""
import copy
import unittest
import bootstrap_v9
from fixtures_v9 import cases,schedule
from runtime_v9 import OperandState
from component_state import ComponentState
from adapter_v7 import collect_raw_input
from tools_v5 import RestrictedTools
from audit_rules import reference_bindings
from evaluate_v7 import evaluate


def prepared(c,intervention=True):
    t=RestrictedTools(c['rows'],c['target']);s=OperandState(t) if intervention else ComponentState(t,'mapping')
    collect_raw_input(s,'B');s.call('get_calculator',{'calculator_id':c['target']['calculator_id']});return s


def args_for(c):
    refs=reference_bindings(c['target']['calculator_id'],c['rows'],c['target']['patient_id'],c['target']['episode_id'])
    bindings=[]
    for k,ref in refs.items():
        ids=[g[0] for g in ref['required_source_groups']]
        values=[c['rows'][int(fid[5:])-1]['value_text'] for fid in ids]
        op='lowercase' if k=='opioid_name' else 'multiply' if len(ids)>1 else 'identity'
        bindings.append({'parameter':k,'value':ref['value'],'operation':op,'operands':[{'fact_id':fid,'value':text if op=='lowercase' else float(text.split()[0])} for fid,text in zip(ids,values)]})
    return {k:v for k,v in c['target'].items() if k!='output_type'}|{'bindings':bindings}


class Design(unittest.TestCase):
    def test_pair_and_reference_paths(self):
        plan=schedule();self.assertEqual(len(plan),16)
        for c in cases():
            pair=[x for x in plan if x['case_id']==c['id']]
            for key in ('prompt','rows','target'):self.assertEqual(pair[0][key],pair[1][key])
            s=prepared(c);r,e=s.call('execute_calculator',args_for(c));self.assertFalse(e,r)
            s.call('submit_result',{'result_id':r['result_id']})
            self.assertTrue(evaluate({'answer':s.tools.answer,'result_ledger':s.results,'trace':[]},c)['preliminary_correct'])

    def test_bad_chains_rejected_without_result(self):
        c=cases()[2];good=args_for(c)
        bads=[]
        a=copy.deepcopy(good);a['bindings'][1]['operands'].pop();bads.append(a)
        a=copy.deepcopy(good);a['bindings'][1]['operands'][0]['value']=999;bads.append(a)
        a=copy.deepcopy(good);a['bindings'][1]['value']=999;bads.append(a)
        a=copy.deepcopy(good);a['bindings'][1]['operands'][0]['fact_id']='fact_9999';bads.append(a)
        old=next((f'fact_{i+1:04d}',float(r['value_text'])) for i,r in enumerate(c['rows']) if r['episode_id']=='old_visit')
        a=copy.deepcopy(good);a['bindings'][1]['operands'][1]={'fact_id':old[0],'value':old[1]};bads.append(a)
        a=copy.deepcopy(good);a['bindings'][0]['operation']='arbitrary';bads.append(a)
        for args in bads:
            s=prepared(c);_,error=s.call('execute_calculator',args);self.assertTrue(error);self.assertEqual(s.results,[])

    def test_coherent_but_incomplete_chain_remains_wrong(self):
        c=cases()[0];a=args_for(c);b=a['bindings'][1]
        b.update(operation='identity',operands=b['operands'][:1],value=b['operands'][0]['value'])
        s=prepared(c);r,e=s.call('execute_calculator',a);self.assertFalse(e)
        s.call('submit_result',{'result_id':r['result_id']})
        self.assertFalse(evaluate({'answer':s.tools.answer,'result_ledger':s.results,'trace':[]},c)['preliminary_correct'])

    def test_identical_initial_raw_state(self):
        for c in cases():
            self.assertEqual(prepared(c).snapshot(),prepared(c,False).snapshot())


if __name__=='__main__':unittest.main()
