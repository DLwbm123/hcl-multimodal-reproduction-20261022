import os,sys,json,tempfile,unittest
from pathlib import Path
from PIL import Image
R=Path(os.environ['RUN_ROOT']);P=json.loads((R/'protocol.json').read_text());sys.path[:0]=[str(R/'code')]+P['python_paths']
from grounding import parse_region,GroundingTools,use_macro,fit
from graph_agent import GraphAgentModel,decode_action,AgentProtocolError,model_observation

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
 def test_transport_names_removed_without_touching_evidence(self):
  a={'artifact':'crop_'+'a'*32+'.png','mask_artifact':'mask_a.png','overlay_artifact':'overlay_a.png','status':'ok','crop_bbox_original':[0,0,64,64],'observation':'Visible tissue; artifact is a medical term.'}
  b={**a,'artifact':'crop_'+'b'*32+'.png','mask_artifact':'mask_b.png','overlay_artifact':'overlay_b.png'}
  for tool in ['inspect_region','segmentation']:
   self.assertEqual(model_observation(tool,json.dumps(a),True),model_observation(tool,json.dumps(b),True));z=json.loads(model_observation(tool,json.dumps(a),True));self.assertEqual(z['observation'],a['observation']);self.assertEqual(z['crop_bbox_original'],a['crop_bbox_original'])
  self.assertEqual(model_observation('answer_visual_question',json.dumps(a),True),json.dumps(a));self.assertEqual(model_observation('inspect_region','not json',True),'not json');self.assertEqual(model_observation('inspect_region',json.dumps(a),False),json.dumps(a))
 def test_real_graph_keeps_private_receipt_but_cleans_both_model_nodes(self):
  class RecordingBackend:
   scope='fixture'
   def __init__(self):self.prompts=[];self.controllers=0
   def generate(self,prompt,*,state):
    phase=state['router_phase'];self.prompts.append((phase,prompt))
    if phase=='graph_controller':
     self.controllers+=1
     return '{"tool":"inspect_region","arguments":{"bbox":[0,0,64,64],"question":"Describe tissue"}}' if self.controllers==1 else '{"finish":true}'
    return 'visible tissue'
  for clean in [False,True]:
   with tempfile.TemporaryDirectory() as d:
    r=Path(d);Image.new('RGB',(64,64),'red').save(r/'input.png');b=RecordingBackend();g=GraphAgentModel(b,r/'trace',tool_runtime=GroundingTools(r/'artifacts'));g.generate('guidance',state={'question':'Describe tissue','task_context':{'files':[{'type':'image','path':str(r/'input.png')}]},'strip_artifact_names':clean})
    e=next(e for e in g.episodes[-1]['events'] if e['node']=='execute');result=json.loads(e['result']);artifact=result['artifact'];self.assertTrue((r/'artifacts'/artifact).exists())
    for phase in ['graph_controller','graph_finalize']:
     prompts=[p for x,p in b.prompts if x==phase];self.assertEqual(any(artifact in p for p in prompts),not clean);self.assertTrue(any('visible tissue' in p for p in prompts))

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
 def test_fullview_control_changes_actual_view_preserves_proposal(self):
  for full,box in [(False,[9,9,55,55]),(True,[0,0,64,64])]:
   with tempfile.TemporaryDirectory() as d:
    r=Path(d);Image.new('RGB',(64,64),'red').save(r/'input.png');backend=Backend('{"present":true,"bbox":[200,200,800,800]}');g=GraphAgentModel(backend,r/'trace',tool_runtime=GroundingTools(r/'artifacts'))
    g.generate('guidance',state={'question':'Where is the polyp?','task_context':{'files':[{'type':'image','path':str(r/'input.png')}]},'grounded_macro':True,'fullview_control':full,'independent_evidence':True})
    events=[e for e in g.episodes[-1]['events'] if e['node']=='execute'];proposal=json.loads(events[0]['result']);result=json.loads(events[1]['result']);self.assertEqual(proposal['crop_bbox_original'],[8,8,56,56]);self.assertEqual(result['crop_bbox_original'],[0,0,64,64] if full else proposal['crop_bbox_original']);self.assertEqual(backend.calls,['graph_tool','graph_tool','graph_finalize'])
 def test_hidden_primitive_and_unseen_intent_cannot_activate(self):
  with tempfile.TemporaryDirectory() as d:
   runtime=GroundingTools(d)
   self.assertNotIn('propose_target_region',runtime.descriptions)
   with self.assertRaises(AgentProtocolError):decode_action('{"tool":"propose_target_region","arguments":{}}',runtime)
  self.assertFalse(use_macro('How many instruments?','macro',{}));self.assertFalse(use_macro('What color?','proposed',{}));self.assertTrue(use_macro('What color?','proposed',{'allowed_macro_intents':['color']}))
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks));save(R/'CPU_CHECK.json',{'passed':result.wasSuccessful(),'tests':result.testsRun,'model_calls':0});assert result.wasSuccessful()
