"""Check task dispatch, fixed-control placement and graph failure accounting."""
import os,sys,json,tempfile,unittest
from pathlib import Path
R=Path(os.environ['RUN_ROOT']);P=json.loads((R/'protocol.json').read_text());sys.path[:0]=[str(R/'code')]+P['python_paths']
from agent_adapter import Dispatch
from check_graph import Backend,Checks
from graph_agent import GraphAgentModel
from analyze import paired
class DispatchChecks(unittest.TestCase):
 def test_vqa_only_and_fixed_guidance(self):
  with tempfile.TemporaryDirectory() as d:
   b=Backend();b.vqa_ids={'v'};g=GraphAgentModel(b,Path(d)/'trace');router=Dispatch(g,b,[{'question':'old question','answer':'fixed_exemplar'}])
   answer=router.generate('visible',state={'question':'How many?','internal_sample_id':'v','answer':'FORBIDDEN_LABEL'})
   self.assertEqual(answer,'2');self.assertEqual(len(g.episodes),1)
   self.assertTrue(any('fixed_exemplar' in prompt for prompt,_ in b.inputs))
   self.assertTrue(all('FORBIDDEN_LABEL' not in prompt and 'answer' not in state for prompt,state in b.inputs))
   router.generate('other',state={'question':'classification','internal_sample_id':'c','router_phase':'graph_tool'})
   self.assertEqual(len(g.episodes),1);self.assertEqual(b.inputs[-1][0],'other')
 def test_paired_denominator(self):
  r=paired([True]*200,[False]*200)
  self.assertEqual(r['gain'],200);self.assertEqual(r['delta_pp'],100);self.assertEqual(r['paired95_pp'],[100,100])
suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in [Checks,DispatchChecks]])
result=unittest.TextTestRunner(verbosity=2).run(suite)
(R/'AGENT_CHECK.json').write_text(json.dumps({'passed':result.wasSuccessful(),'tests':result.testsRun,'scripted_only':True,'confirmation_labels_opened':False},indent=2))
assert result.wasSuccessful()
