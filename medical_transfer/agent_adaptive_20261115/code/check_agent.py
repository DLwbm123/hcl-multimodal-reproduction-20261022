import os,sys,json,tempfile,unittest
from pathlib import Path
R=Path(os.environ['RUN_ROOT']);P=json.loads((R/'protocol.json').read_text());sys.path[:0]=[str(R/'code')]+P['python_paths']
from graph_agent import GraphAgentModel,AgentProtocolError,decode_action
class Backend:
 scope='fixture'
 def __init__(self,repeat=False,invalid=False):self.inputs=[];self.repeat=repeat;self.invalid=invalid
 def generate(self,prompt,*,state):
  self.inputs.append((prompt,state));phase=state['router_phase']
  if phase=='graph_controller':
   if self.invalid:return '{}'
   if self.repeat or 'OBSERVATIONS:\n[]' in prompt:return '{"tool":"answer_visual_question","arguments":{"instruction":"Inspect current image"}}'
   return '{"finish":true}'
  return '2'
class Checks(unittest.TestCase):
 def test_flow_and_role_isolation(self):
  with tempfile.TemporaryDirectory() as d:
   b=Backend();g=GraphAgentModel(b,Path(d)/'trace');ans=g.generate('PRIVATE_GUIDANCE',state={'question':'Visible question','answer':'SECRET_LABEL','task_name':'SECRET_TASK','task_context':{'files':[]}})
   self.assertEqual(ans,'2');self.assertEqual([x['node'] for x in g.episodes[-1]['events']],['process','execute','process','finalize'])
   self.assertGreater(g.episodes[-1]['checkpoint_count'],2)
   for prompt,state in b.inputs:
    self.assertNotIn('SECRET_LABEL',prompt);self.assertNotIn('answer',state);self.assertNotIn('task_name',state)
    if state['router_phase']=='graph_controller':self.assertNotIn('PRIVATE_GUIDANCE',prompt)
    else:self.assertIn('PRIVATE_GUIDANCE',prompt)
 def test_bound_then_real_finalize(self):
  with tempfile.TemporaryDirectory() as d:
   g=GraphAgentModel(Backend(repeat=True),Path(d)/'trace');self.assertEqual(g.generate('g',state={'question':'q'}),'2')
   self.assertEqual(g.episodes[-1]['tool_calls'],2);self.assertEqual(g.episodes[-1]['controller_calls'],2)
   self.assertEqual(g.episodes[-1]['events'][-2]['node'],'budget_finalize');self.assertEqual(g.episodes[-1]['events'][-1]['node'],'finalize')
 def test_invalid_is_not_success(self):
  with tempfile.TemporaryDirectory() as d:
   g=GraphAgentModel(Backend(invalid=True),Path(d)/'trace');self.assertEqual(g.generate('g',state={'question':'q'}),'');self.assertEqual(g.episodes[-1]['status'],'failed')
   self.assertFalse(any(x['node']=='execute' for x in g.episodes[-1]['events']))
 def test_no_untrusted_tool_or_early_finish(self):
  for raw in ['{"tool":"shell","arguments":{}}','{"finish":false}','{"answer":"x"}','```bad```']:
   with self.assertRaises(AgentProtocolError):decode_action(raw)
 def test_budget_exception_propagates(self):
  class F(Backend):
   def generate(self,*a,**k):raise RuntimeError('budget')
  with tempfile.TemporaryDirectory() as d:
   g=GraphAgentModel(F(),Path(d)/'trace')
   with self.assertRaises(RuntimeError):g.generate('g',state={'question':'q'})
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
(R/'AGENT_CHECK.json').write_text(json.dumps({'passed':result.wasSuccessful(),'tests':result.testsRun,'scripted_only':True,'confirmation_labels_opened':False},indent=2));assert result.wasSuccessful()
