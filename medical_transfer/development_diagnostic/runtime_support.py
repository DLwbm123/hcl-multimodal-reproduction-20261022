"""Frozen run-only corrections and audit wrappers; no change to historical code."""
import json,math,time,os
from pathlib import Path

def canonical_bbox(raw,question):
    if 'output only one json object with bbox' not in question.lower():return raw
    import re
    text=raw.strip()
    fence=re.fullmatch(r"```(?:json)?\s*(.*?)\s*```",text,flags=re.IGNORECASE|re.DOTALL)
    if fence:text=fence.group(1)
    try:value=json.loads(text)
    except (ValueError,TypeError):return raw
    item=value[0] if isinstance(value,list) and len(value)==1 else value
    if not isinstance(item,dict) or not set(item).issubset({'bbox','bbox_2d','label'}):return raw
    if ('bbox' in item)==('bbox_2d' in item):return raw
    box=item.get('bbox_2d',item.get('bbox'))
    if not isinstance(box,list) or len(box)!=4:return raw
    if not all(isinstance(x,(float,int)) and not isinstance(x,bool) and math.isfinite(x) for x in box):return raw
    if box[2]<=box[0] or box[3]<=box[1]:return raw
    return json.dumps({'bbox':box})

def build_model(out,budget):
    import torch
    from hcl.models import LocalTransformersModel,_state_phase
    class AuditedModel(LocalTransformersModel):
        scope='preflight'
        def generate(self,prompt,*,state):
            # No backend retry can bypass the shared attempt ledger.
            if time.time()-budget['started_epoch']>=23.5*3600:raise RuntimeError('shared original deadline exhausted')
            count_file=out/'model_attempts.json'
            count=json.loads(count_file.read_text())['attempts'] if count_file.exists() else 0
            if count>=budget['per_worker_call_cap']:raise RuntimeError('worker call budget exhausted')
            tmp=count_file.with_suffix('.tmp');tmp.write_text(json.dumps({'attempts':count+1,'original_started_epoch':budget['started_epoch']}));tmp.replace(count_file)
            t=time.monotonic();phase=_state_phase(state);answer=None;ok=False
            try:
                answer=super().generate(prompt,state=state);ok=True
                normalized=canonical_bbox(answer,str(state.get('question','')))
                if normalized!=answer:
                    with (out/'private_schema_normalization.jsonl').open('a') as f:f.write(json.dumps({'sample_id':state.get('internal_sample_id'),'raw':answer,'canonical':normalized})+'\n')
                return normalized
            finally:
                e={'attempt':count+1,'scope':self.scope,'phase':phase,'success':ok,'seconds':time.monotonic()-t,'output_chars':len(answer or ''),'image_supplied':bool(self._image_path_from_state(state)),'peak_allocated_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0}
                with (out/'call_costs.jsonl').open('a') as f:f.write(json.dumps(e)+'\n')
    model=AuditedModel(name='Qwen2.5-VL-7B-Instruct',path=budget['model_path'],torch_dtype='bfloat16',attn_implementation='sdpa',max_new_tokens=2048,answer_max_new_tokens=512,temperature=0.0,enable_thinking=False,local_files_only=True,log_path=None)
    return model

def install_protocol(pipeline,model,out,plan):
    from types import MethodType
    from hcl.runtime_fail_closed import install_memory_selector_fail_closed
    from hcl.pipeline import HCLPipeline
    install_memory_selector_fail_closed(HCLPipeline)
    # Upstream continual_train invokes this even when the probe flag is false.
    # Never build correctness-conditioned probe sets from the held-out test.
    def no_probe(self,*args,**kwargs):
        if self.forgetting_probe.enabled:raise RuntimeError('test probe forbidden')
        return None
    pipeline._build_probe_set=MethodType(no_probe,pipeline)
    cfg=pipeline.router_config
    cfg.use_chat_tool_calls=False;cfg.tool_auto_select=False
    cfg.workflow_selector_max_new_tokens=512
    cfg.skill_selector_max_new_tokens=256;cfg.skill_selector_fallback_to_empty_on_invalid=True
    cfg.memory_selector_max_new_tokens=256;cfg.memory_selector_retry_max_new_tokens=256
    pipeline.task_interface_config.max_new_tokens=640
    pipeline.task_interface_config.retry_max_new_tokens=1024
    pipeline.task_interface_config.fallback_to_rules_on_invalid=True
    pipeline.memory.config.abstract_max_new_tokens=512
    pipeline.memory.abstract.max_new_tokens=512
    pipeline.memory.config.anchor_capacity_per_task=plan['anchors_per_task']
    pipeline.memory.anchor.capacity_per_task=plan['anchors_per_task']
    # scope also covers candidate-specific routers and read-only anchor evaluation.
    original_predict=pipeline._predict_with_router
    def scoped(self,*args,**kwargs):
        previous=model.scope;model.scope=kwargs.get('progress_label','unknown')
        try:return original_predict(*args,**kwargs)
        finally:model.scope=previous
    pipeline._predict_with_router=MethodType(scoped,pipeline)
    original_test=pipeline.test
    def readonly(self,*args,**kwargs):
        result=original_test(*args,**kwargs)
        with (out/'read_only_checks.jsonl').open('a') as f:f.write(json.dumps({'split':kwargs.get('split'),'learning_state_unchanged':True,'examples':result.evaluation['metrics']['total']})+'\n')
        return result
    pipeline.test=MethodType(readonly,pipeline)
