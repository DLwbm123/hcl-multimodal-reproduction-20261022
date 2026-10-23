import unittest,json
from runtime_state import ExecutionState
from clinical_catalog import search,execute,details
from tools import COLUMNS,calculate

def row(patient='p',ep='e',field='x',value='1'):
 d={k:'' for k in COLUMNS};d.update(patient_id=patient,episode_id=ep,field_name=field,value_text=value);return d

def state():
 s=ExecutionState([row(),row(ep='previous'),row(patient='other',ep='other')]);s.call('read_facts',{});return s

def args(ep='e',fact='fact_0001'):
 params={'heart_rate':90,'systolic_bp':120,'patient_type':'adult'}
 return {'calculator_id':'shock_index','episode_id':ep,'patient_id':'p','parameters':params,'input_sources':{k:{'fact_ids':[fact],'transformation':'Synthetic typed input fixture'} for k in params}}

class InterfaceTests(unittest.TestCase):
 def test_exact_alias(self):
  self.assertEqual(search('Heart Rate/Systolic Blood Pressure Ratio')['calculators'][0]['calculator_id'],'shock_index')
  self.assertEqual(search('qSOFA')['calculators'][0]['calculator_id'],'qsofa_score')
 def test_optional_not_all_defaults(self):
  d=details('shock_index')['parameters'];self.assertNotIn('diastolic_bp',d['required']);self.assertIn('patient_type',d['required'])
  self.assertAlmostEqual(execute('shock_index',{'heart_rate':90,'systolic_bp':120,'patient_type':'adult'})['value'],.75)
  with self.assertRaises(ValueError):execute('shock_index',{'heart_rate':90,'systolic_bp':120})
 def test_conditional_parameters(self):
  p={'age':40,'cancer':False,'chronic_cardiopulmonary_disease':False}
  with self.assertRaises(ValueError):execute('spesi',p)
  self.assertEqual(execute('spesi',{**p,'heart_rate_gte_110':False,'sbp_lt_100':False,'spo2_lt_90':False})['value'],0)
 def test_explicit_reference_submission(self):
  s=state();out,err=s.call('execute_calculator',args());self.assertFalse(err);self.assertIsNone(s.tools.answer)
  self.assertFalse(s.call('submit_answer',{'calculators':[{'result_id':out['result_id']}]})[1])
  self.assertEqual(s.tools.answer['calculators'][0]['episode_id'],'e');self.assertAlmostEqual(s.tools.answer['calculators'][0]['value'],.75)
 def test_wrong_source_patient_rejected(self):
  s=state();self.assertTrue(s.call('execute_calculator',args(fact='fact_0003'))[1]);self.assertEqual(s.results,[])
 def test_cross_episode_sources_retained(self):
  s=state();self.assertFalse(s.call('execute_calculator',args(fact='fact_0002'))[1]);r=s.results[0]
  self.assertEqual(r['episode_id'],'e');self.assertEqual(r['input_bindings']['heart_rate']['facts'][0]['episode_id'],'previous')
 def test_unknown_scope_and_rebinding(self):
  s=state();self.assertTrue(s.call('execute_calculator',args(ep='missing'))[1]);o,e=s.call('execute_calculator',args());self.assertFalse(e)
  self.assertTrue(s.call('submit_answer',{'calculators':[{'result_id':o['result_id'],'episode_id':'previous'}]})[1]);self.assertIsNone(s.tools.answer)
 def test_repeated_errors_are_errors(self):
  s=state();a={'calculator_id':'does_not_exist'};o,e=s.call('get_calculator',a);self.assertTrue(e)
  short,e=s.call('get_calculator',a);self.assertTrue(e);self.assertTrue(short['cached'])
  full,e=s.call('get_observation',{'observation_id':o['observation_id']});self.assertTrue(e);self.assertEqual(full['error'],'ValueError')
 def test_search_does_not_reset_progress(self):
  s=state();s.call('search_calculators',{'query':'shock'});n=s.no_progress;s.call('search_calculators',{'query':'heart'});self.assertGreater(s.no_progress,n)
  s.call('execute_calculator',args());self.assertEqual(s.no_progress,0)
 def test_arithmetic_grammar(self):
  self.assertEqual(calculate('1 if x > 2 else 0',{'x':3}),1)
  with self.assertRaises(SyntaxError):calculate('if(x>2,1,0)',{'x':3})
 def test_request_limit_includes_cache(self):
  s=state()
  for _ in range(80):s.call('read_facts',{})
  self.assertEqual(s.call('read_facts',{})[0]['error'],'tool_call_limit');self.assertIsNone(s.tools.answer)
 def test_instance_isolation(self):
  a=state();b=state();a.call('execute_calculator',args());self.assertEqual(b.results,[])
 def test_context_tool_correlation(self):
  from adapter import load_upstream,adapter_class
  from langchain_core.messages import AIMessage,ToolMessage,SystemMessage,HumanMessage
  from qwen_transport import to_qwen
  upstream=load_upstream(__import__('os').environ['MEDICALHARNESS_REPO']);H=adapter_class(upstream);h=H.__new__(H);h.execution=state();h.preamble=[SystemMessage(content='s'),HumanMessage(content='task')];h.history=[];h.recalls=[]
  for i in range(7):
   h.history.extend([{'role':'ai','message':AIMessage(content='',tool_calls=[{'id':str(i),'name':'read_facts','args':{}}])},{'role':'tool','tool_name':'read_facts','message':ToolMessage(content='{}',name='read_facts',tool_call_id=str(i))}])
  h.visible_history=lambda:h.history
  converted=to_qwen(h.build_view());self.assertEqual(sum(x['role']=='function' for x in converted),3)
if __name__=='__main__':unittest.main()
