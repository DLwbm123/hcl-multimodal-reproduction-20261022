"""Stage A and CPU contract checks; no model generated outputs."""
import copy
import json
import unittest
from pathlib import Path
import bootstrap_v10
from runtime_v10 import RepairState,in_scope
from fixtures_v10 import recovery_cases,confirmation_cases,schedule
from tools_v5 import RestrictedTools
from adapter_v7 import collect_raw_input
from audit_rules import reference_bindings,Unresolved
from evaluate_v10 import evaluate
from evaluate_v7 import evaluate as old_evaluate


def state(c,precise=True,limit=12):
    s=RepairState(RestrictedTools(c['rows'],c['target']),precise,limit);collect_raw_input(s,'B');return s


def reference_call(c):
    refs=reference_bindings(c['target']['calculator_id'],c['rows'],c['target']['patient_id'],c['target']['episode_id']);bindings=[]
    for key,ref in refs.items():
        ids=[g[0] for g in ref['required_source_groups']];operation='lowercase' if key=='opioid_name' else 'multiply' if len(ids)>1 else 'identity'
        operands=[{'fact_id':fid,'value':c['rows'][int(fid[5:])-1]['value_text'] if operation=='lowercase' else float(c['rows'][int(fid[5:])-1]['value_text'].split()[0])} for fid in ids]
        bindings.append({'parameter':key,'value':ref['value'],'operation':operation,'operands':operands})
    return {k:v for k,v in c['target'].items() if k!='output_type'}|{'bindings':bindings}


def score_state(s,c):return evaluate({'answer':s.tools.answer,'result_ledger':s.results,'execution_state':s.snapshot(),'trace':[]},c)


class Design(unittest.TestCase):
    def test_new_reference_paths_and_unsolvable(self):
        for c in recovery_cases()+confirmation_cases():
            if c['expected_outcome']=='answerable':
                s=state(c);out,err=s.call('execute_calculator',reference_call(c));self.assertFalse(err,out)
                s.call('submit_result',{'result_id':out['result_id']});self.assertTrue(score_state(s,c)['preliminary_correct'])
            else:
                with self.assertRaises(Unresolved):reference_call(c)

    def test_all_seed_errors_and_independent_coverage(self):
        for c in recovery_cases():
            s=state(c);out,err=s.call('execute_calculator',c['seed_call']);self.assertTrue(err);self.assertEqual(s.results,[]);self.assertFalse(s.cache.keys() & {'execute_calculator:'+json.dumps(c['seed_call'],sort_keys=True)})
            self.assertIn('errors',out)
        c=recovery_cases()[0];s=state(c);out,_=s.call('execute_calculator',c['seed_call'])
        self.assertEqual(out['errors'][0]['parameter'],'opioid_name');self.assertEqual(out['binding_checks'][1]['status'],'locally_consistent')
        c=recovery_cases()[2];out,_=state(c).call('execute_calculator',c['seed_call']);self.assertTrue(any(e['code']=='SCHEMA_ERROR' for e in out['errors']));self.assertEqual(out['binding_checks'][0]['status'],'locally_consistent');self.assertFalse(out['binding_checks'][0]['schema_valid'])
        c=recovery_cases()[3];out,_=state(c).call('execute_calculator',c['seed_call']);self.assertTrue(any(e['code']=='SOURCE_SCOPE_MISMATCH' for e in out['errors']))

    def test_boundaries_budget_cache_and_repairs(self):
        c=recovery_cases()[0];good=reference_call(c);bads=[]
        for change in ('source','scope','math','missing','duplicate'):
            a=copy.deepcopy(good)
            if change=='source':a['bindings'][1]['operands'][0]['value']=999
            if change=='scope':a['patient_id']='other'
            if change=='math':a['bindings'][1]['value']=999
            if change=='missing':a['bindings'][1]['operands'].pop()
            if change=='duplicate':a['bindings'].append(copy.deepcopy(a['bindings'][0]))
            bads.append(a)
        for a in bads:
            s=state(c);_,e=s.call('execute_calculator',a);self.assertTrue(e);self.assertEqual(s.results,[])
            _,e=s.call('execute_calculator',good);self.assertFalse(e)
        s=state(c,limit=2);self.assertEqual(s.snapshot()['remaining_turns'],2);s.prepare(1);self.assertEqual(s.snapshot()['remaining_turns'],1)
        out,_=s.call('execute_calculator',c['seed_call']);out2,_=s.call('execute_calculator',c['seed_call']);self.assertNotEqual(out['observation_id'],out2['observation_id']);self.assertTrue(s.repair_exhausted)
        out,e=s.call('execute_calculator',good);self.assertTrue(e);self.assertEqual(s.results,[])
        # Truly cross-patient and cross-episode sources, not just invocation mismatch.
        for key in ('patient_id','episode_id'):
            mutated=copy.deepcopy(c);mutated['rows'][0][key]='outside';s=state(mutated);out,e=s.call('execute_calculator',good);self.assertTrue(e);self.assertEqual(s.results,[])
        t={'calculator_id':'corrected_qt','patient_id':'p','episode_id':'now'};f={'field_name':'sex','patient_id':'p','episode_id':'old'}
        self.assertTrue(in_scope(t,'sex',f));self.assertFalse(in_scope(t,'qt',f));self.assertFalse(in_scope(t,'sex',{**f,'patient_id':'other'}))

    def test_counterfactual_v9(self):
        # Read-only historical traces; changes exist only in copied diagnostic calls.
        old=Path(__file__).resolve().parent.parent/'operand_binding_v9'
        receipts=[]
        for cid in ['M01_I','M02_I','M04_I','M05_I','M03_I','M06_I']:
            r=json.loads((old/'episodes'/cid/'RESULT.json').read_text())
            c={'rows':r['initial_execution_state']['evidence_state']['read_facts'],'target':r['initial_execution_state']['target'],'stage':'mapping','expected_outcome':'answerable'}
            a=copy.deepcopy(next(x['args'] for event in r['trace'] for x in event.get('tool_calls',[]) if x['name']=='execute_calculator'))
            b=next(b for b in a['bindings'] if b['parameter']=='opioid_name');b['operation']='lowercase'
            if cid=='M03_I':b['operands'][0]['value']='Oxycodone'
            s=state(c);out,error=s.call('execute_calculator',a);self.assertFalse(error,out)
            s.call('submit_result',{'result_id':out['result_id']});correct=score_state(s,c)['preliminary_correct']
            self.assertEqual(correct,cid not in ('M03_I','M06_I'))
            receipts.append({'id':cid,'diagnostic_only_not_model_success':True,'changed_only_opioid_binding':True,'trusted_correct':correct,'result':s.results})
        r=json.loads((old/'episodes/M08_I/RESULT.json').read_text())
        c={'rows':r['initial_execution_state']['evidence_state']['read_facts'],'target':r['initial_execution_state']['target']}
        bad_schema=json.loads(next(x['args'] for e in r['trace'] for x in e.get('invalid_tool_calls',[])))
        bad_scope=next(x['args'] for e in r['trace'] for x in e.get('tool_calls',[]) if x['name']=='execute_calculator')
        s=state(c);a=s.diagnose(bad_schema);b=s.diagnose(bad_scope)
        self.assertTrue(any(e['code']=='SCHEMA_ERROR' for e in a['errors']))
        self.assertTrue(any(e['code']=='SOURCE_SCOPE_MISMATCH' for e in a['errors']))
        self.assertTrue(any(e['code']=='SOURCE_SCOPE_MISMATCH' for e in b['errors']))
        self.assertEqual(s.results,[])
        receipts.append({'id':'M08_I','diagnostic_only_not_model_success':True,'schema_attempt':a,'scope_attempt':b})
        (Path(__file__).parent/'COUNTERFACTUAL_A.json').write_text(json.dumps({'real_model_calls':0,'historical_records_unchanged':True,'replays':receipts},indent=2))

    def test_actual_adapter_serializes_schema_diagnostics(self):
        import asyncio,tempfile
        from adapter_v5 import load_upstream
        from adapter_v10 import adapter_class
        from langchain_core.messages import AIMessage
        from qwen_transport import to_qwen
        root=Path(__file__).parent;config=json.loads((root/'runtime_config.private.json').read_text())
        upstream=load_upstream(config['package'])
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            out=Path(tmp);prompt=out/'prompt.txt';prompt.write_text('CPU test only')
            args=upstream.parse_args(['--workspace',str(out),'--prompt-file',str(prompt),'--model','cpu-fixture','--upstream','http://127.0.0.1:1/v1','--planning','off','--context-policy','none','--verification','off','--timeout-seconds','300'])
            c=next(c for c in schedule('B') if c['id']=='B03_B')
            h=adapter_class(upstream)(args,'http://127.0.0.1:1/v1',out,upstream.EventLog(out),{},tools=RestrictedTools(c['rows'],c['target']),model=object(),workflow=(),case=c)
            asyncio.run(h.setup_tools(None));self.assertEqual(h.execution.snapshot()['remaining_turns'],2)
            h.history.append({'event_id':h.log.emit('fixture'), 'role':'ai','message':AIMessage(content='',invalid_tool_calls=[{'name':'execute_calculator','args':json.dumps(c['seed_call']),'id':'cpu-invalid','error':'schema fixture'}])})
            asyncio.run(h.node_dispatch({'turn':1}))
            exposed=json.loads(h.history[-1]['message'].content)
            self.assertEqual(exposed['errors'][0]['parameter'],'sodium');self.assertIn('binding_checks',exposed)
            self.assertEqual(h.execution.results,[]);self.assertTrue(h.execution.repair_exhausted)
            self.assertTrue(to_qwen(h.build_view()))

    def test_pair_design(self):
        plan=schedule('B');self.assertEqual(len(plan),32);self.assertEqual(len(schedule('C','B')),24)
        for case in recovery_cases():
            copies=[c for c in plan if c['case_id']==case['id']];self.assertEqual({c['arm'] for c in copies},set('ABCD'))
            for k in ('rows','target','prompt','seed_call'):self.assertTrue(all(c[k]==copies[0][k] for c in copies))


if __name__=='__main__':unittest.main()
