"""Opt-in candidate structural repair. No global patches or evaluation threshold changes."""
import copy
import re
from hcl.prompt_templates import PromptTemplate,_PLACEHOLDER
from hcl.optimizer import Optimizer,_artifact_validation_errors

FIELDS={
    'task_interface_structuring':frozenset({'raw_visible_input'}),
    'router_workflow':frozenset({'model_visible_input','selected_skills'}),
    'memory_selector':frozenset({'current_input','workflow_decision','memory_candidates','retrieval_top_k'}),
    'router_context':frozenset({'model_visible_input','workflow_decision','selected_memory','selected_skills','selected_tools'}),
}
KNOWN_UNAVAILABLE=frozenset({'historical_anchor_metrics','skill_selection_rationale','memory_selection_rationale'})
HEADINGS={'Historical anchor metrics:','Skill selection rationale:','Memory selection rationale:'}
SAFETY=('In raw memory, correct_feedback is authoritative supervision; previous_attempt may be incorrect. '
        'Never copy an incorrect previous_attempt; apply the lesson to the current visible task.')
FINAL='Reason internally and output only the final answer in the requested format.'

def fields(section):
    # Match the pinned renderer's literal-brace protection, not str.format syntax.
    protected=section.replace('{{','\x00LEFT\x00').replace('}}','\x00RIGHT\x00')
    return {m.group(1) for m in _PLACEHOLDER.finditer(protected)}

def validate(component,content):
    if component not in FIELDS:return ['unsupported_component']
    errors=list(_artifact_validation_errors(component,content))
    sections=content.get('sections',[])
    if not isinstance(sections,list) or not all(isinstance(s,str) for s in sections):return ['invalid_sections']
    used=set().union(*(fields(s) for s in sections)) if sections else set()
    errors += ['unsupported_runtime_field:'+name for name in sorted(used-FIELDS[component])]
    errors += ['missing_runtime_field:'+name for name in sorted(FIELDS[component]-used)]
    # Each component gets ONLY its real caller values, unlike upstream's union fixture.
    try:PromptTemplate.from_dict(content).render({name:'[]' if name in {'selected_skills','selected_memory','selected_tools','memory_candidates'} else '0' if name=='retrieval_top_k' else '{}' for name in FIELDS[component]})
    except (ValueError,KeyError,TypeError) as exc:errors.append('component_render_failure:'+str(exc))
    return sorted(set(errors))

def repair(component,content):
    original=copy.deepcopy(content);fixed=copy.deepcopy(content);actions=[]
    if component not in FIELDS:return original,{'actions':[],'errors':['unsupported_component']}
    sections=fixed.get('sections')
    if not isinstance(sections,list) or not all(isinstance(s,str) for s in sections):
        return original,{'actions':[],'errors':['invalid_sections']}
    # Repair only the three observed unsupported runtime inputs, never arbitrary new fields.
    bad={name for s in sections for name in fields(s)-FIELDS[component]}
    if bad-KNOWN_UNAVAILABLE:
        return original,{'actions':[],'errors':validate(component,original)}
    if bad:
        kept=[]
        for index,section in enumerate(sections):
            if fields(section)&bad:
                # Never discard a mixed legal/illegal section: repair would erase live inputs.
                if fields(section)&FIELDS[component]:
                    return original,{'actions':[],'errors':['mixed_supported_and_unsupported_section']}
                actions.append({'action':'drop_unavailable_input_section','index':index,'fields':sorted(fields(section)&bad)})
            elif section.strip() in HEADINGS or any(name.replace('_',' ') in section.lower() for name in bad):
                actions.append({'action':'drop_unavailable_dependency_prose','index':index})
            else:kept.append(section)
        kept.append('Runtime input boundary: use only these supplied runtime fields: '+', '.join(sorted(FIELDS[component]))+'. '
                    'Historical anchor metrics and separate skill/memory-selection rationales are not supplied; '
                    'do not infer, request, or fabricate them.')
        fixed['sections']=kept
    if component=='router_context':
        text='\n'.join(fixed['sections']).lower()
        if not all(t in text for t in ['correct_feedback','previous_attempt']) or not any(t in text for t in ['may be incorrect','can be incorrect']):
            fixed['sections'].append(SAFETY);actions.append({'action':'restore_canonical_memory_safety_rule'})
        if not any(t in text for t in ['final answer only','output only the final']):
            fixed['sections'].append(FINAL);actions.append({'action':'restore_final_only_rule'})
    errors=validate(component,fixed)
    return fixed,{'actions':actions,'errors':errors}

class ContractOptimizer(Optimizer):
    """Select explicitly in a fresh future pipeline. Existing workers never import this."""
    def generate_artifact_candidates(self,component,train_result,*,model,current_artifact=None,harness_version=0,generation_context=None):
        if component not in FIELDS:raise ValueError('unsupported component')
        context=copy.deepcopy(generation_context or {})
        context['immutable_template_contract']={
            'allowed_runtime_fields':sorted(FIELDS[component]),
            'required_runtime_fields':sorted(FIELDS[component]),
            'no_new_placeholders':True,'do_not_request_evaluator_inputs':True,
            'router_context_required_literal_rule':SAFETY if component=='router_context' else None,
            'rule':'Preserve runtime variable names exactly. No extra retries or replacement candidates.'}
        return super().generate_artifact_candidates(component,train_result,model=model,
            current_artifact=current_artifact,harness_version=harness_version,generation_context=context)

    def _normalize_candidates(self,raw_candidates,**kwargs):
        candidates,rejections=super()._normalize_candidates(raw_candidates,**kwargs)
        for candidate in candidates:
            original=copy.deepcopy(candidate['artifact_content'])
            fixed,receipt=repair(kwargs['component'],original)
            candidate['original_artifact_content']=original
            candidate['artifact_content']=fixed
            candidate['structural_repair']=receipt
            candidate['validation_errors']=receipt['errors']
        return candidates,rejections


def install_candidate_contract(pipeline):
    """Explicitly install on a fresh pipeline only; never hot-patch a running experiment."""
    from dataclasses import fields as dataclass_fields
    if getattr(pipeline,'_candidate_template_contract_installed',False):raise ValueError('already installed')
    if pipeline.checkpoint.has_state or pipeline.global_batch_index or pipeline.harness_version:
        raise ValueError('fresh pipeline required')
    if pipeline.optimizer.history or pipeline.optimizer.last_generation_diagnostics:
        raise ValueError('optimizer already used')
    if any(int(v.get('version',0)) for v in pipeline.component_versions.values()):
        raise ValueError('existing learned component state')
    original=pipeline.optimizer
    for path in [original.candidate_history_path,original.generation_history_path]:
        if path.exists() and path.stat().st_size:raise ValueError('nonempty optimizer history on disk')
    options={field.name:copy.deepcopy(getattr(original,field.name)) for field in dataclass_fields(original) if field.init}
    pipeline.optimizer=ContractOptimizer(**options)
    pipeline._candidate_template_contract_installed=True
    return pipeline
