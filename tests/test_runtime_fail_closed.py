import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from hcl.cli import _build_pipeline
from hcl.models import RuleBasedSmokeModel
from hcl.pipeline import HCLPipeline
from hcl.runtime_fail_closed import install_memory_selector_fail_closed

class FailedSampleAccountingTest(unittest.TestCase):
    def test_invalid_selector_is_wrong_not_successful_fallback_and_other_errors_propagate(self):
        original=HCLPipeline._predict_one
        try:
            install_memory_selector_fail_closed(HCLPipeline)
            with tempfile.TemporaryDirectory() as d:
                root=Path(d)
                p=_build_pipeline({'storage_dir':str(root),'task_interface':{'mode':'rule'},'router':{'use_llm_workflow_selector':False,'use_memory_selector':False,'use_skill_selector':False,'use_tool_selector':False},'memory':{'abstract_mode':'rule','skill_enabled':False}},model=RuleBasedSmokeModel(),base_dir=root)
                x={'task_id':'math/train/one','task_name':'math','task_type':'gsm8k','question':'2+2','answer':'4'}
                error='Memory Selector LLM output remained invalid after 2 attempt(s): selected_1_role_invalid'
                with patch('hcl.router.Router.run',side_effect=ValueError(error)):
                    preds=p._predict([x],update_memory=True,progress_label='train')
                result=p.evaluator.evaluate([x],preds)['metrics']
                self.assertEqual(result['total'],1)
                self.assertEqual(result['correct'],0)
                self.assertEqual(result['format_compliance_rate'],0)
                self.assertTrue(preds[0]['trace']['intermediate_failure']['answering_skipped'])
                self.assertEqual(p.memory.summary()['raw'],1)
                before=p.memory.snapshot()
                with patch('hcl.router.Router.run',side_effect=ValueError(error)):
                    p._predict([x],update_memory=False,progress_label='test')
                self.assertEqual(before,p.memory.snapshot())
                with patch('hcl.router.Router.run',side_effect=ValueError('unrelated bug')):
                    with self.assertRaisesRegex(ValueError,'unrelated bug'):
                        p._predict([x],update_memory=False)
        finally:
            HCLPipeline._predict_one=original
