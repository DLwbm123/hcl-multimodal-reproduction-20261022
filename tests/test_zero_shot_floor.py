import json
import unittest

from hcl.optimizer import Optimizer
from hcl.router import Router, RouterConfig
from hcl.task_interface import TaskInterface, TaskInterfaceConfig


class ZeroShotFloorTests(unittest.TestCase):
    def test_workflow_can_choose_original_answer_without_memory(self):
        class Model:
            def generate(self, prompt, *, state):
                self_test.assertIn('"raw_direct"', prompt)
                self_test.assertEqual(state['router_phase'], 'plan_workflow')
                return json.dumps({'workflow_name': 'raw_direct'})

        self_test = self
        config = TaskInterfaceConfig(mode='rule')
        chunk = TaskInterface(config).build_taskinterfacechunk({'task_id': '1', 'question': 'What color?'}, 0)
        router = Router(Model(), RouterConfig(use_skill_selector=False))
        result = router.run(chunk, config, raw_answer=lambda: {'answer':'green','trace':{}})
        self.assertEqual(result['answer'], 'green')
        self.assertEqual(result['trace']['route'], 'raw_direct')

    def test_below_floor_is_rejected_even_if_current_version_is_worse(self):
        optimizer = Optimizer(selection_objective='plasticity', historical_loss_budget=0,
                              min_primary_score_delta=1e-9)
        candidate = {'candidate_id':'c'}
        current = {'metrics':{'primary_score':.2,'correct':2}, 'zero_shot_floor':.5}
        def result(score):
            return {'metrics': {'primary_score':score,'correct':int(score*10),
                                'format_compliance_rate':1.0},
                    'historical_anchor_metrics':{'forget_count':0}}
        self.assertIsNone(optimizer.select_best_candidate([candidate], {'c':result(.4)}, current))
        self.assertIn('zero_shot_floor_not_met', optimizer.history[-1]['candidate_rejections'][0]['reasons'])
        self.assertEqual(optimizer.select_best_candidate([candidate], {'c':result(.5)}, current), candidate)


if __name__=='__main__':
    unittest.main()
