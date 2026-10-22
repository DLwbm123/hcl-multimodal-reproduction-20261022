"""Read visible validation records only. Never joins private answers or invokes a backend."""
import copy,json,ast
from pathlib import Path
from template_contract import repair,validate,FIELDS,ContractOptimizer,install_candidate_contract
from hcl.task_interface import TaskInterface,TaskInterfaceConfig
from hcl.router import Router,RouterConfig
from hcl.prompt_templates import PromptTemplate

class PromptCapture:
    def __init__(self):self.prompts=[]
    def generate(self,prompt,*,state):
        self.prompts.append(prompt)
        if state.get('router_phase')!='plan_workflow':raise AssertionError('unexpected stub role')
        return json.dumps({'workflow_name':'direct','steps':['observe','answer'],'use_memory':False,'use_tools':False,'memory_query':'','tool_needs':[],'rationale':'CPU fixture'})

def verify_runtime_keys(upstream):
    tree=ast.parse((Path(upstream)/'hcl/router.py').read_text())
    expected={'plan_workflow':FIELDS['router_workflow'],'build_context':FIELDS['router_context'],'select_memory':FIELDS['memory_selector']}
    checked=[]
    for node in ast.walk(tree):
        if not isinstance(node,ast.FunctionDef) or node.name not in expected:continue
        bindings=[n.value for n in ast.walk(node) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='prompt_values' for t in n.targets) and isinstance(n.value,ast.Dict)]
        if len(bindings)!=1:raise AssertionError('runtime prompt_values structure changed')
        actual={k.value for k in bindings[0].keys if isinstance(k,ast.Constant)}
        assert actual==expected[node.name],(node.name,actual,expected[node.name])
        checked.append(node.name)
    assert len(checked)==3
    return checked

def run(project,upstream,out):
    project=Path(project);out=Path(out);out.mkdir(exist_ok=True)
    candidates=json.loads((out/'source_artifacts.json').read_text())
    rows=[]
    for task in ['classification','polyp_localization','visual_qa']:
        f=project/'prepared_data/visible'/task/'validation.jsonl'
        source=sorted((json.loads(line) for line in f.read_text().splitlines()),key=lambda r:r['id'])
        selected=source[:4]
        for row in selected:
            rows.append({**row,'task_id':row['id'],'task_name':task,'task_type':'medical_'+task,'split':'validation'})
    assert len(rows)==12 and all('answer' not in r and 'metadata' not in r for r in rows)
    records=[];repaired=[];renders=0
    for c in candidates:
        component=c['component'];original=copy.deepcopy(c['artifact_content'])
        fixed,receipt=repair(component,original)
        assert original==c['artifact_content']
        assert not receipt['errors'],(c['candidate_name'],receipt)
        assert repair(component,fixed)[0]==fixed
        if not receipt['actions']:assert fixed==original
        repaired.append({**c,'artifact_content':fixed,'structural_repair':receipt})
        for e in rows:
            capture=PromptCapture();ti=TaskInterface(TaskInterfaceConfig(mode='rule'),model=capture)
            chunk=ti.build_taskinterfacechunk(e,0)
            if component=='task_interface_structuring':
                custom=TaskInterface(TaskInterfaceConfig(mode='rule'),model=capture,structuring_template_content=fixed)
                text=custom._render_structuring_prompt(e)
            elif component=='router_workflow':
                router=Router(capture,RouterConfig(capability_enabled=False),workflow_template_content=fixed)
                router.plan_workflow(chunk,selected_skills=[])
                assert len(capture.prompts)==1
                text=capture.prompts[0]
            elif component=='router_context':
                router=Router(capture,RouterConfig(capability_enabled=False),context_template_content=fixed)
                text=router.build_context(chunk,workflow_decision={'use_memory':False},selected_memory=[],selected_skills=[],selected_tools=[])
            else:
                # Match the real selector call's component-specific runtime map; no anchors/gold.
                text=PromptTemplate.from_dict(fixed).render({'current_input':json.dumps(TaskInterface.to_model_visible(chunk)),
                    'workflow_decision':'{}','memory_candidates':'[]','retrieval_top_k':2})
            assert text and all('{'+name+'}' not in text for name in FIELDS[component])
            assert 'PRIVATE_GOLD_CANARY' not in text
            renders+=1
        records.append({'component':component,'name':c['candidate_name'],'before_errors':validate(component,original),
            'after_errors':receipt['errors'],'actions':receipt['actions'],'development_renders':len(rows)})
    assert len(candidates)==24
    assert sum(bool(r['actions']) for r in records)==4
    # Exercise actual Optimizer normalization, preserving count and original versions/IDs.
    import tempfile
    with tempfile.TemporaryDirectory(dir=out) as temp:
        import sys
        sys.path.insert(0,str(project/'src'))
        from medical_hcl.bridge import construct_pipeline
        pipe=construct_pipeline(project/'prepared_data',Path(temp)/'pipeline')
        install_candidate_contract(pipe)
        opt=pipe.optimizer
        assert pipe.model.calls==0
        assert (opt.candidate_count,opt.historical_loss_budget,opt.min_primary_score_delta,opt.min_format_compliance_rate)==(3,0,1e-9,1.)
        for c in candidates:
            normalized,rejections=opt._normalize_candidates([{'candidate_name':c['candidate_name'],'artifact_content':c['artifact_content']}],component=c['component'],source_version=0,candidate_version=c['candidate_version'],count=3,current_artifact={})
            assert len(normalized)==1 and not rejections and not normalized[0]['validation_errors']
    (out/'repaired_artifacts.json').write_text(json.dumps(repaired,indent=2))
    (out/'private_development_selection.json').write_text(json.dumps({'role':'validation','case_ids':[r['task_id'] for r in rows],'selection':'first four ascending visible IDs per task'},indent=2))
    report={'passed':True,'candidate_artifacts':24,'known_bad_repaired':4,'valid_controls_unchanged':20,
        'development_cases':12,'development_renders':renders,'fresh_medical_pipeline_hook_checked':True,'runtime_key_functions_checked':verify_runtime_keys(upstream),
        'model_inference_calls':0,'training_or_continual_learning_runs':0,'test_or_anchor_records_loaded':0,
        'private_answer_files_loaded':0,'current_experiment_modified':False,'comparison_started':False,
        'scope':'template structural/render acceptance only; no performance improvement or coordinate-fix claim',
        'candidates':records}
    (out/'development_acceptance.json').write_text(json.dumps(report,indent=2))
    return report
