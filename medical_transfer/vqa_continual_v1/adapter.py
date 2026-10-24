"""Opt-in shared-task stream adapter; stage metadata is evaluator-only."""
import copy,json,re
from contextlib import contextmanager
from types import MethodType

ORDER=('A','B','C')

def earlier_stage_anchors(anchors,train):
    stages={e['metadata']['stream_stage'] for e in train}
    assert len(stages)==1 and all(e['task_name']=='visual_qa' for e in train)
    stage=next(iter(stages));cut=ORDER.index(stage)
    chosen=[e for e in anchors if ORDER.index(e['metadata']['stream_stage'])<cut]
    assert all(e['metadata']['role']=='retention_anchor' and e['task_name']=='visual_qa' for e in chosen)
    assert not {e['task_id'] for e in chosen}&{e['task_id'] for e in train}
    return chosen

@contextmanager
def stage_anchor_scope():
    # The upstream task-level exclusion would remove every same-task VQA anchor.
    # Patch only inside this serial worker's train call; restore even on failure.
    import hcl.pipeline as module
    old=module._historical_anchors_for_train
    module._historical_anchors_for_train=earlier_stage_anchors
    try:yield
    finally:module._historical_anchors_for_train=old

def freeze_anchors(pipe,examples):
    assert len(examples)<=72 and len({e['task_id'] for e in examples})==len(examples)
    assert all(e['metadata']['role']=='retention_anchor' for e in examples)
    # Already independently sampled and group-deduplicated: keep exact stage coverage.
    pipe.memory.anchor.records=copy.deepcopy(examples)
    pipe.memory.anchor._persist()

def fixed_select(self,*,current_input,workflow_decision,**kwargs):
    from hcl.memory import _memory_candidate_text
    tok=lambda x:set(re.findall(r'[a-z0-9]+',x.lower()))
    query=tok(json.dumps(current_input,sort_keys=True))
    pool=self.retrieval_candidates(current_input=current_input,workflow_decision=workflow_decision)
    def rank(item):
        terms=tok(_memory_candidate_text(item));union=query|terms
        return (-(len(query&terms)/len(union) if union else 0),str(item['memory_id']))
    selected=[{**x,'selected_role':'support','selection_reason':'fixed lexical overlap','selection_risk':'medium'} for x in sorted(pool,key=rank)[:self.config.retrieval_top_k]]
    return selected,{'selected':[{'memory_id':x['memory_id']} for x in selected],'candidate_count':len(pool),'policy':'fixed_lexical_top2'}

def install_stream(pipe,arm,plan):
    assert not pipe.checkpoint.has_state and not pipe.memory.raw.records
    pipe.memory.config.skill_enabled=False
    pipe.router_config.use_skill_selector=False
    pipe.optimizer.components=['memory_selector']
    pipe.optimizer.candidate_count=3
    pipe.optimizer.selection_objective='plasticity'
    pipe.optimizer.commit_policy='plasticity_first'
    pipe.optimizer.min_primary_score_delta=1e-9
    pipe.optimizer.min_correct_gain=1
    pipe.optimizer.min_format_compliance_rate=1.0
    pipe.optimizer.historical_loss_budget=0 if arm=='stability' else None
    if arm=='fixed_memory':pipe.memory.select_for_context=MethodType(fixed_select,pipe.memory)
    assert pipe.memory.config.raw_capacity_per_task==10 and pipe.memory.config.abstract_capacity_per_task==2
    assert pipe.memory.config.retrieval_top_k==2

def snapshot(pipe):
    return copy.deepcopy((pipe.memory.snapshot(),pipe.component_versions,pipe.harness_version,pipe.optimizer.history,pipe.router_config.to_dict(),pipe.task_interface_config.to_dict(),pipe.checkpoint.state))
