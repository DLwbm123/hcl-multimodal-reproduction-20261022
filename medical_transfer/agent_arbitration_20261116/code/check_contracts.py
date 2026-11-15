import os,sys,json,tempfile,unittest
from pathlib import Path
R=Path(os.environ['RUN_ROOT']);P=json.loads((R/'protocol.json').read_text());sys.path[:0]=[str(R/'code')]+P['python_paths']
from contracts import compile_contracts,contract_for,gate
from graph_agent import GraphAgentModel
class B:
 scope='fixture'
 def __init__(self):self.inputs=[]
 def generate(self,prompt,*,state):
  self.inputs.append((prompt,state));return '{"tool":"answer_visual_question","arguments":{"instruction":"BIASED_GUESS"}}' if state['router_phase']=='graph_controller' else 'red'
class Checks(unittest.TestCase):
 def test_training_boundary_and_query_scope(self):
  train=[{'split':'train','task_name':'visual_qa','question':'What color is the abnormality?','answer':'red; pink'} for _ in range(48)];a=compile_contracts(train);self.assertEqual(a['vocabulary']['color'],['pink','red']);self.assertEqual(contract_for('How many instruments?',a),'')
  train[0]['split']='confirmation'
  with self.assertRaises(AssertionError):compile_contracts(train)
 def test_independent_tool_and_final_guidance(self):
  with tempfile.TemporaryDirectory() as d:
   b=B();g=GraphAgentModel(b,Path(d)/'trace');g.generate('PRIVATE_HISTORY',state={'question':'What color is the abnormality?','evidence_contract':'OUTPUT_SCHEMA','independent_evidence':True})
   for text,state in b.inputs:
    if state['router_phase']=='graph_tool':self.assertNotIn('PRIVATE_HISTORY',text);self.assertNotIn('BIASED_GUESS',text);self.assertIn('OUTPUT_SCHEMA',text)
    if state['router_phase']=='graph_finalize':self.assertIn('PRIVATE_HISTORY',text);self.assertIn('OUTPUT_SCHEMA',text)
 def test_contract_ablation_keeps_original_tool_guidance(self):
  with tempfile.TemporaryDirectory() as d:
   b=B();g=GraphAgentModel(b,Path(d)/'trace');g.generate('PRIVATE_HISTORY',state={'question':'q','evidence_contract':'OUTPUT_SCHEMA'})
   text=next(t for t,s in b.inputs if s['router_phase']=='graph_tool');self.assertIn('PRIVATE_HISTORY',text);self.assertIn('BIASED_GUESS',text);self.assertIn('OUTPUT_SCHEMA',text)
 def test_gate_no_fallback_or_relaxation(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d)
   def write(loss,gain):
    for role in ['eval_base','eval_evidence']:
     p=r/role/'private_scores.jsonl';p.parent.mkdir(exist_ok=True)
     a=[{'task_id':str(i),'success':i<4+(gain if role=='eval_evidence' else 0)} for i in range(48)];h=[{'task_id':str(i),'success':i<10 and not(role=='eval_evidence' and i<loss)} for i in range(72)]
     p.write_text(''.join(json.dumps(x)+'\n' for x in [{'split':'selection','case_results':a,'metrics':{'format_compliance_rate':1.}},{'split':'historical','case_results':h,'metrics':{}}]))
   write(2,1);self.assertTrue(gate(r)['accepted']);write(3,1);self.assertFalse(gate(r)['accepted']);write(0,0);self.assertFalse(gate(r)['accepted'])
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks));(R/'CPU_CHECK.json').write_text(json.dumps({'passed':result.wasSuccessful(),'tests':result.testsRun,'confirmation_used':False}));assert result.wasSuccessful()
