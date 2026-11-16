import os,sys,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as NS
R=Path(os.environ['RUN_ROOT']);P=json.loads((R/'protocol.json').read_text());sys.path[:0]=[str(R/'code')]+P['python_paths']
from arbitration import fit,gate,strategy
from agent_adapter import Dispatch

def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))
def lines(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(x)+'\n' for x in v))
class Checks(unittest.TestCase):
 def test_fit_requires_gain_and_no_losses_without_reading_validation(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);save(r/'protocol.json',P);save(r/'INCUMBENT_POLICY.json',{'allowed_intents':['color','location']});train=[{'task_id':str(i),'question':'How many instruments?' if i<24 else 'What color is the polyp?','split':'train','answer':'unused'} for i in range(48)];save(r/'private_learning.json',{'train':train})
   for role in ['incumbent','contract']:
    ok=lambda i:(i<2 or i==24) if role=='incumbent' else i<5
    lines(r/('fit_'+role)/'private_scores.jsonl',[{'split':'counterfactual_train','metrics':{},'case_results':[{'task_id':str(i),'success':ok(i)} for i in range(48)]}])
   x=fit(r);self.assertEqual(x['contract_override_intents'],['count']);self.assertEqual(x['training_groups']['color']['losses'],1)
   train[0]['split']='confirmation';save(r/'private_learning.json',{'train':train})
   with self.assertRaises(AssertionError):fit(r)
 def test_empty_override_preserves_incumbent(self):
  old={'allowed_intents':['color','location']}
  self.assertEqual(strategy('What color?','proposed',old,{}),'evidence');self.assertEqual(strategy('How many?','proposed',old,{}),'base');self.assertEqual(strategy('What color?','contract',old,{}),'contract')
 def test_dispatch_three_source_boundaries(self):
  with tempfile.TemporaryDirectory() as d:
   graph=NS(trace_path=Path(d)/'trace',task_contracts={'vocabulary':{'color':['red'],'location':['top']}},source_arm='proposed',incumbent_policy={'allowed_intents':['color','location']},source_policy={'contract_override_intents':['color']});graph.generate=lambda prompt,**kw:kw['state']
   boundary=NS(vqa_ids={'q'},scope='fixture',backend=NS(_image_path_from_state=lambda state:'fixture.png'));pipe=NS(memory=NS(raw=NS(records=[])));dispatch=Dispatch(graph,boundary,pipe,{},None,None,False)
   for q,independent,contract in [('What color is the polyp?',False,True),('Where is the polyp?',True,True),('How many instruments?',False,False)]:
    value=dispatch.generate('guidance',state={'internal_sample_id':'q','question':q});self.assertEqual(value['independent_evidence'],independent);self.assertEqual(bool(value['evidence_contract']),contract)
 def test_gate_does_not_accept_cost_only_or_excess_history_loss(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d)
   def write(gain,loss):
    for role in ['incumbent','proposed']:
     a=[{'task_id':str(i),'success':i<4+(gain if role=='proposed' else 0)} for i in range(24)];h=[{'task_id':str(i),'success':i<10 and not(role=='proposed' and i<loss)} for i in range(72)]
     lines(r/('eval_'+role)/'private_scores.jsonl',[{'split':'selection','case_results':a,'metrics':{'format_compliance_rate':1.}},{'split':'historical','case_results':h,'metrics':{}}])
   write(0,0);self.assertFalse(gate(r)['accepted']);write(1,2);self.assertTrue(gate(r)['accepted']);write(1,3);self.assertFalse(gate(r)['accepted'])
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks));save(R/'CPU_CHECK.json',{'passed':result.wasSuccessful(),'tests':result.testsRun,'model_calls':0});assert result.wasSuccessful()
