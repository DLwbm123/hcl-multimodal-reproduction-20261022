import unittest,json
from diagnose import source_assistance
class SourceAssistanceTests(unittest.TestCase):
 def test_assistance_excludes_parameter_values_and_score(self):
  a={'calculator_id':'synthetic','patient_id':'p','episode_id':'e','parameters':{'x':93812},'input_sources':{'x':{'fact_ids':['fact_0001'],'transformation':'reference mapping'}},'computed_score':92813}
  text=source_assistance(a)
  self.assertNotIn('93812',text);self.assertNotIn('92813',text)
  self.assertIn('fact_0001',text);self.assertIn('input_sources',text)
  self.assertEqual(a['parameters']['x'],93812)
