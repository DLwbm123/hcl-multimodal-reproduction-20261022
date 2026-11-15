import os,sys,json,tempfile,unittest
from pathlib import Path
from PIL import Image
R=Path(os.environ['RUN_ROOT']);P=json.loads((R/'protocol.json').read_text());sys.path[:0]=[str(R/'code')]+P['python_paths']
from grounding import parse_region,GroundingTools,use_macro,fit
from graph_agent import GraphAgentModel,decode_action,AgentProtocolError

def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))
def lines(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(x)+'\n' for x in v))
class Backend:
 scope='fixture'
 def __init__(self,proposal):self.proposal=proposal;self.calls=[]
 def generate(self,prompt,*,state):
  self.calls.append(state['router_phase'])
  if state['router_phase']=='graph_finalize':return 'red'
  if 'Propose the smallest' in prompt:return self.proposal
  return 'white'
class Checks(unittest.TestCase):
 def test_geometry_mapping_and_invalid_bounds(self):
  x=parse_region('{"present":true,"bbox":[250,250,750,750]}',(100,200));self.assertEqual(x['target_bbox_original'],[25,50,75,150]);self.assertEqual(x['crop_bbox_original'],[20,40,80,160])
  y=parse_region('{"present":true,"bbox":[499,499,501,501]}',(64,64));self.assertGreaterEqual(y['crop_bbox_original'][2]-y['crop_bbox_original'][0],28)
  for v in [{'present':True,'bbox':[-1,0,100,100]},{'present':True,'bbox':[0,0,1001,1000]},{'present':True,'bbox':[False,0,100,100]},{'present':False,'bbox':[0,0,100,100]}]:
   with self.assertRaises(AssertionError):parse_region(json.dumps(v),(100,100))
 def test_macro_two_observations_then_real_finalize(self):
  for proposal,expected in [('{"present":true,"bbox":[200,200,800,800]}','inspect_region'),('not json','answer_visual_question'),('{"present":false,"bbox":null}','answer_visual_question')]:
   with tempfile.TemporaryDirectory() as d:
    r=Path(d);Image.new('RGB',(64,64),'red').save(r/'input.png');backend=Backend(proposal);runtime=GroundingTools(r/'artifacts');g=GraphAgentModel(backend,r/'trace',tool_runtime=runtime)
    ans=g.generate('old guidance',state={'question':'What color is the polyp?','task_context':{'files':[{'type':'image','path':str(r/'input.png')}]},'grounded_macro':True,'macro_decision_source':'training_probe'})
    self.assertEqual(ans,'red');self.assertEqual(backend.calls,['graph_tool','graph_tool','graph_finalize']);self.assertEqual(g.episodes[-1]['tool_calls'],2);self.assertEqual(g.episodes[-1]['controller_calls'],0);self.assertEqual([e['tool'] for e in g.episodes[-1]['events'] if e['node']=='execute'],['propose_target_region',expected])
 def test_hidden_primitive_and_unseen_intent_cannot_activate(self):
  with tempfile.TemporaryDirectory() as d:
   runtime=GroundingTools(d)
   self.assertNotIn('propose_target_region',runtime.descriptions)
   with self.assertRaises(AgentProtocolError):decode_action('{"tool":"propose_target_region","arguments":{}}',runtime)
  self.assertFalse(use_macro('How many instruments?','macro',{}));self.assertFalse(use_macro('What color?','proposed',{}));self.assertTrue(use_macro('What color?','proposed',{'allowed_macro_intents':['color']}))
 def test_fit_only_training_and_eligible_gain_without_loss(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);save(r/'protocol.json',P);save(r/'INCUMBENT_POLICY.json',{'allowed_intents':['color','location']});train=[{'task_id':str(i),'split':'train','question':('What color?' if i<16 else 'Where is the polyp?' if i<32 else 'Describe scene')} for i in range(48)];save(r/'private_learning.json',{'train':train})
   for arm,correct in [('incumbent',{0,16,32}),('macro',{0,1,32,33})]:lines(r/('fit_'+arm)/'private_scores.jsonl',[{'split':'counterfactual_train','metrics':{},'case_results':[{'task_id':str(i),'success':i in correct} for i in range(48)]}])
   self.assertEqual(fit(r)['allowed_macro_intents'],['color']);train[0]['split']='confirmation';save(r/'private_learning.json',{'train':train})
   with self.assertRaises(AssertionError):fit(r)
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks));save(R/'CPU_CHECK.json',{'passed':result.wasSuccessful(),'tests':result.testsRun,'model_calls':0});assert result.wasSuccessful()
