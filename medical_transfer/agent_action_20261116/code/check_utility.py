import os,sys,json,tempfile,unittest
from pathlib import Path
R=Path(os.environ['RUN_ROOT']);P=json.loads((R/'protocol.json').read_text());sys.path[:0]=[str(R/'code')]+P['python_paths']
from utility import observation_key,fit,gate
from graph_agent import GraphAgentModel

def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x))
def lines(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(a)+'\n' for a in x))
class B:
 scope='fixture'
 def __init__(self):self.calls=[]
 def generate(self,prompt,*,state):
  self.calls.append(state['router_phase'])
  return '{"tool":"answer_visual_question","arguments":{"instruction":"Count current image"}}' if state['router_phase']=='graph_controller' else '2'
class Checks(unittest.TestCase):
 def test_no_answer_identity_or_uncertain_stop(self):
  self.assertEqual(observation_key('How many instruments?','answer_visual_question','2'),observation_key('How many instruments?','answer_visual_question','9'))
  for tool,result in [('localize_region','2'),('answer_visual_question','unclear'),('answer_visual_question','cannot determine')]:self.assertIsNone(observation_key('How many instruments?',tool,result))
 def test_first_observation_then_real_finalize(self):
  with tempfile.TemporaryDirectory() as d:
   b=B();g=GraphAgentModel(b,Path(d)/'trace');k=observation_key('How many instruments?','answer_visual_question','2')
   self.assertEqual(g.generate('guidance',state={'question':'How many instruments?','utility_policy':{'allowed_keys':[k]}}),'2')
   self.assertEqual(b.calls,['graph_controller','graph_tool','graph_finalize']);self.assertEqual(g.episodes[-1]['tool_calls'],1);self.assertEqual(g.episodes[-1]['events'][-2]['node'],'learned_stop')
 def test_unseen_feature_keeps_full_loop(self):
  with tempfile.TemporaryDirectory() as d:
   b=B();g=GraphAgentModel(b,Path(d)/'trace');g.generate('g',state={'question':'How many instruments?','utility_policy':{'allowed_keys':['presence|answer_visual_question|binary']}});self.assertEqual(g.episodes[-1]['tool_calls'],2)
 def test_training_losses_and_mismatched_observation_abstain(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);save(r/'protocol.json',P);save(r/'private_learning.json',{'train':[{'task_id':str(i),'question':'How many instruments?'} for i in range(48)]})
   for role,name in [('fit_full','full'),('fit_one','one')]:
    lines(r/role/'private_scores.jsonl',[{'split':'counterfactual_train','case_results':[{'task_id':str(i),'success':i==0 and role=='fit_full'} for i in range(48)],'metrics':{}}])
    lines(r/role/'private_graph_traces.jsonl',[{'sample_id':str(i),'scope':'counterfactual_train:'+name,'events':[{'node':'execute','tool':'answer_visual_question','args':{'instruction':'count'},'result':'2' if i or role=='fit_full' else '3'}]} for i in range(48)])
   x=fit(r);self.assertEqual(x['excluded']['unmatched_first_observation'],1);self.assertEqual(len(x['allowed_keys']),1)
   p=r/'fit_one/private_graph_traces.jsonl';p.write_text(p.read_text().replace('"result": "3"','"result": "2"'));self.assertEqual(fit(r)['allowed_keys'],[])
 def test_gate_keeps_original_gain_and_history_limits(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d)
   def write(loss,gain):
    for role in ['fit_full','candidate']:
     a=[{'task_id':str(i),'success':i<4+(gain if role=='candidate' else 0)} for i in range(24)]
     h=[{'task_id':str(i),'success':i<10 and not(role=='candidate' and i<loss)} for i in range(72)]
     lines(r/role/'private_scores.jsonl',[{'split':'selection','case_results':a,'metrics':{'format_compliance_rate':1.}},{'split':'historical','case_results':h,'metrics':{}}])
   write(2,1);self.assertTrue(gate(r)['accepted']);write(3,1);self.assertFalse(gate(r)['accepted']);write(0,0);self.assertFalse(gate(r)['accepted'])
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks));(R/'CPU_CHECK.json').write_text(json.dumps({'passed':result.wasSuccessful(),'tests':result.testsRun,'confirmation_labels_opened':False}));assert result.wasSuccessful()
