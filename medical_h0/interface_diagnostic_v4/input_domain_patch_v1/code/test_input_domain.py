import unittest
from unittest.mock import patch
from clinical_catalog import details,execute,CALCS

class DomainTests(unittest.TestCase):
 def inputs(self,category):
  return dict(burden_of_illness=category,no_hypotension=True,no_copd=True,solid_tumor_or_no_fungal_hx=True,no_dehydration=True,outpatient_status=True,age_lt_60=True)
 def test_all_documented_categories_preserve_scores(self):
  for category,score in [('none_mild',26),('moderate',24),('severe',21)]:
   self.assertEqual(execute('mascc_score',self.inputs(category))['value'],score)
 def test_unknown_categories_never_reach_formula(self):
  with patch.object(CALCS['mascc_score'],'calculate',side_effect=AssertionError('formula must not execute')):
   for value in ['None or mild','unknown','SEVERE','']:
    with self.assertRaises(ValueError):execute('mascc_score',self.inputs(value))
 def test_visible_definition_exposes_same_domain(self):
  self.assertEqual(details('mascc_score')['parameters']['properties']['burden_of_illness']['enum'],['none_mild','moderate','severe'])
