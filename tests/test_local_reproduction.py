import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from hcl.cli import _build_pipeline
from hcl.models import RuleBasedSmokeModel

class ReproductionStateTests(unittest.TestCase):
 def pipeline(self, root):
  return _build_pipeline({'storage_dir':str(root),'task_interface':{'mode':'rule'},'router':{'use_llm_workflow_selector':False,'use_memory_selector':False,'use_skill_selector':False,'use_tool_selector':False},'memory':{'abstract_mode':'rule','skill_enabled':False},'optimizer':{'components':[]}},model=RuleBasedSmokeModel(),base_dir=root)
 def test_final_test_does_not_write_learning_state(self):
  with tempfile.TemporaryDirectory() as d:
   p=self.pipeline(Path(d)); before=p.memory.snapshot(); versions=p._component_version_snapshot()
   p.test([{'task_id':'heldout','task_name':'math','task_type':'gsm8k','question':'2+2','answer':'4'}],split='test')
   self.assertEqual(before,p.memory.snapshot()); self.assertEqual(versions,p._component_version_snapshot())
 def test_rejected_memory_candidate_restores_all_memory(self):
  with tempfile.TemporaryDirectory() as d:
   p=self.pipeline(Path(d));before=p.memory.snapshot()
   sample={'task_id':'train/math/one','task_name':'math','task_type':'gsm8k','question':'2+2','answer':'4','split':'train'}
   with patch.object(p,'_memory_candidate_decision',return_value=(False,['forced_regression_test'])),patch.object(p,'_select_batch_anchors',return_value={}):
    p._train_batch_and_optimize([sample],val_examples=[sample],current_val_examples=[sample],historical_anchor_examples=[],anchor_context={'required':False},split='train',batch_index=0,batch_count=1)
   self.assertEqual(before,p.memory.snapshot())
 def test_retention_counts_losses_despite_recoveries(self):
  from hcl.evaluator import ExactMatchEvaluator
  xs=[{'task_id':'a','answer':'yes'},{'task_id':'b','answer':'yes'}]
  result=ExactMatchEvaluator().evaluate(xs,[{'task_id':'a','answer':'no'},{'task_id':'b','answer':'yes'}],reference_predictions=[{'task_id':'a','answer':'yes'},{'task_id':'b','answer':'no'}])
  self.assertEqual(result['metrics']['forget_count'],1)
  self.assertEqual(result['metrics']['recovered_count'],1)

 def test_failed_candidate_execution_restores_learning_memory(self):
  with tempfile.TemporaryDirectory() as d:
   p=self.pipeline(Path(d));before=p.memory.snapshot()
   sample={'task_id':'train/math/one','task_name':'math','task_type':'gsm8k','question':'2+2','answer':'4','split':'train'}
   def fail(*args,**kwargs):
    p.memory.raw.records.append({'partial_candidate_write':True})
    p.memory.raw._persist()
    raise ValueError('invalid selector schema')
   with patch.object(p,'_evaluate_train_batch',side_effect=fail):
    with self.assertRaisesRegex(ValueError,'invalid selector schema'):
     p._train_batch_and_optimize([sample],val_examples=[sample],current_val_examples=[sample],historical_anchor_examples=[],anchor_context={'required':False},split='train',batch_index=0,batch_count=1)
   self.assertEqual(before,p.memory.snapshot())
