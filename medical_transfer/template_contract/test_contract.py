import copy,tempfile,unittest,json
from pathlib import Path
from template_contract import repair,validate,fields,ContractOptimizer,SAFETY,install_candidate_contract
from hcl.prompt_templates import PromptTemplate
from hcl.models import RuleBasedSmokeModel

UP=Path(__import__('hcl').__file__).resolve().parent

def baseline(component):
    file={'router_workflow':'router/workflow_prompt_default.json','router_context':'router/context_prompt_default.json',
    'task_interface_structuring':'task_interface/structuring_prompt_default.json','memory_selector':'router/memory_selector_prompt_default.json'}[component]
    return json.loads((UP/file).read_text())

class ContractTests(unittest.TestCase):
    def test_known_variables_removed_without_value_injection(self):
        for name in ['historical_anchor_metrics','skill_selection_rationale','memory_selection_rationale']:
            content=baseline('router_workflow');content['sections']+=['Unavailable input:','{'+name+'}']
            fixed,receipt=repair('router_workflow',content)
            self.assertEqual(receipt['errors'],[])
            rendered=PromptTemplate.from_dict(fixed).render({'model_visible_input':'VISIBLE_ONLY','selected_skills':'[]'})
            self.assertNotIn('{'+name+'}',rendered)
            self.assertEqual(content['sections'][-1],'{'+name+'}')
    def test_unknown_or_mixed_fields_fail_closed(self):
        c=baseline('router_workflow');c['sections']+=['{gold_answer}']
        fixed,receipt=repair('router_workflow',c)
        self.assertEqual(fixed,c);self.assertTrue(receipt['errors'])
        c=baseline('router_workflow');c['sections']+=['{model_visible_input} {historical_anchor_metrics}']
        self.assertIn('mixed_supported_and_unsupported_section',repair('router_workflow',c)[1]['errors'])
    def test_global_fixture_field_not_legal_for_workflow(self):
        c=baseline('router_workflow');c['sections']+=['{selected_memory}']
        self.assertIn('unsupported_runtime_field:selected_memory',validate('router_workflow',c))
    def test_exact_memory_keys_restored(self):
        c=baseline('router_context');c['sections']=[s.replace('correct_feedback','Correct feedback').replace('previous_attempt','previous attempt') for s in c['sections']]
        fixed,receipt=repair('router_context',c)
        self.assertIn(SAFETY,fixed['sections']);self.assertEqual(receipt['errors'],[])
    def test_idempotent_baselines_and_json_braces(self):
        for component in ['router_workflow','router_context','memory_selector','task_interface_structuring']:
            c=baseline(component);fixed,receipt=repair(component,c)
            self.assertEqual(fixed,c);self.assertEqual(receipt['errors'],[])
        self.assertEqual(fields('{{unknown}} {known} {"bbox":[1,2,3,4]}'),{'known'})
    def test_missing_required_and_visible_reasoning_stay_invalid(self):
        c=baseline('router_context');c['sections']=[s for s in c['sections'] if s!='{selected_memory}']
        self.assertTrue(repair('router_context',c)[1]['errors'])
        c=baseline('router_context');c['sections'].append('Show your work.')
        self.assertTrue(repair('router_context',c)[1]['errors'])
    def test_explicit_fresh_pipeline_install_preserves_all_optimizer_options(self):
        from types import SimpleNamespace
        from dataclasses import fields as dataclass_fields
        from hcl.optimizer import Optimizer
        with tempfile.TemporaryDirectory() as d:
            opt=Optimizer(record_dir=d,candidate_count=3,historical_loss_budget=0,min_primary_score_delta=1e-9,min_format_compliance_rate=1.)
            before={f.name:copy.deepcopy(getattr(opt,f.name)) for f in dataclass_fields(opt) if f.init}
            pipe=SimpleNamespace(optimizer=opt,checkpoint=SimpleNamespace(has_state=False),global_batch_index=0,harness_version=0,component_versions={'router_workflow':{'version':0}})
            install_candidate_contract(pipe)
            self.assertIsInstance(pipe.optimizer,ContractOptimizer)
            self.assertEqual(before,{f.name:getattr(pipe.optimizer,f.name) for f in dataclass_fields(opt) if f.init})
            with self.assertRaises(ValueError):install_candidate_contract(pipe)
            pipe._candidate_template_contract_installed=False;pipe.checkpoint.has_state=True
            with self.assertRaises(ValueError):install_candidate_contract(pipe)

    def test_optimizer_hook_no_extra_candidate_or_retry_and_gates_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            opt=ContractOptimizer(record_dir=d,candidate_count=3,historical_loss_budget=0,
                min_primary_score_delta=1e-9,min_format_compliance_rate=1.0)
            c=baseline('router_workflow');c['sections']+=['{historical_anchor_metrics}']
            raw=[{'candidate_name':'repair','sections':c['sections']},{'candidate_name':'valid','sections':baseline('router_workflow')['sections']}]
            normalized,rejections=opt._normalize_candidates(raw,component='router_workflow',source_version=0,candidate_version=1,count=3,current_artifact={'content':baseline('router_workflow')})
            self.assertEqual(len(normalized),2);self.assertEqual(rejections,[])
            self.assertTrue(normalized[0]['structural_repair']['actions']);self.assertEqual(normalized[0]['validation_errors'],[])
            self.assertEqual((opt.historical_loss_budget,opt.min_primary_score_delta,opt.min_format_compliance_rate),(0,1e-9,1.0))
            class Capture(RuleBasedSmokeModel):
                calls=0
                def generate(self,prompt,*,state):
                    self.calls+=1
                    if 'allowed_runtime_fields' not in prompt:raise AssertionError('missing generation boundary')
                    return super().generate(prompt,state=state)
            model=Capture()
            opt.generate_artifact_candidates('router_workflow',{},model=model,current_artifact={'content':baseline('router_workflow')})
            self.assertEqual(model.calls,1)

if __name__=='__main__':unittest.main()
