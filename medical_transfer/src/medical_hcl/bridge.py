"""Explicit adapter to the pinned HCL implementation. No backend constructor."""
import copy
import json
from pathlib import Path
from .data import StrictMedicalLoader
from .evaluation import HCLMedicalEvaluator,score_case

class NoInferenceModel:
    name='preparation-only-no-inference'
    calls=0
    def generate(self,*args,**kwargs):
        self.calls+=1
        raise PermissionError('真实模型推理未授权；准备阶段禁止生成')

def construct_pipeline(prepared,storage,*,model=None,real_execution_authorized=False,arm='stability_hcl'):
    if arm not in ('stability_hcl','static_harness'):raise ValueError('first-stage arms only')
    if model is not None and not real_execution_authorized:raise PermissionError('explicit future real execution authorization required')
    # Imports are deliberately lazy and point to the pinned archive selected by caller.
    from hcl.pipeline import HCLPipeline
    from hcl.memory import ExperienceMemory,MemoryConfig
    from hcl.memory.raw import RawMemory,_stratified_interactions,_memory_id,_source
    from hcl.optimizer import Optimizer
    from hcl.evaluator import EvaluationRecorder
    from hcl.router import RouterConfig
    from hcl.task_interface import TaskInterfaceConfig
    root=Path(storage);root.mkdir(parents=True,exist_ok=True)
    vocabulary=json.loads((Path(prepared)/'class_vocabulary.json').read_text())
    evaluator=HCLMedicalEvaluator(vocabulary)
    config=MemoryConfig(record_dir=str(root/'memory'),enabled=arm!='static_harness',raw_capacity_per_task=10,abstract_capacity_per_task=2,anchor_capacity_per_task=64,anchor_selection_mode='split',anchor_identity_mode='visible_input',skill_enabled=True,reset_on_start=False,abstract_mode='llm')

    class MedicalRawMemory(RawMemory):
        def add(self,task_name,examples,predictions,*,anchor_ids=None,replace_anchors=False):
            if anchor_ids:raise ValueError('evaluator-only anchors cannot enter raw memory')
            by={str(p['task_id']):p for p in predictions}
            retained=[r for r in self.records if _source(r,'task_name')!=task_name]
            current={str(_source(r,'task_id')):r for r in self.records if _source(r,'task_name')==task_name}
            for e in examples:
                if e['metadata']['role']!='h_search':raise ValueError('only H-search feedback may be learned')
                key=e['task_id'];p=by.get(key,{})
                answer=p.get('answer') if p.get('status','ok')=='ok' and not p.get('error') else None
                valid,success,_,_=score_case(e,answer,vocabulary)
                current[key]={'memory_id':_memory_id(key),'memory_type':'raw','internal_source':{'task_name':task_name,'task_type':e['task_type'],'task_id':key,'split':e['split']},'question':e['question'],'response':answer,'feedback':e['answer'],'response_is_correct':bool(success),'format_compliant':bool(valid),'output_form':'bbox' if task_name=='polyp_localization' else 'text','is_anchor':False}
            self.records=retained+_stratified_interactions(list(current.values()),self.capacity_per_task)
            self._persist()

    memory=ExperienceMemory(config,model=model or NoInferenceModel())
    memory.raw=MedicalRawMemory(root/'memory/raw_memory.jsonl',capacity_per_task=10)
    optimizer=Optimizer(record_dir=str(root/'optimizer'),candidate_count=3,components=['task_interface_structuring','router_workflow','memory_selector','router_context'],selection_objective='plasticity',commit_policy='plasticity_first',historical_loss_budget=0,max_forget_count=0,min_primary_score_delta=1e-9,min_correct_gain=0,min_format_compliance_rate=1.0,screening_ratio=0)

    class MedicalPipeline(HCLPipeline):
        def train(self,train_data,*,val_data,**kwargs):
            if arm=='static_harness':raise PermissionError('Static cannot learn or search H')
            train=self.dataset_loader.load(train_data);val=self.dataset_loader.load(val_data) if val_data is not None else []
            if not train or not val:raise ValueError('nonempty H-search and validation required')
            if any(e['metadata']['role']!='h_search' for e in train):raise ValueError('learning role violation')
            if any(e['metadata']['role']!='validation' for e in val):raise ValueError('validation role violation')
            return super().train(train,val_data=val,**kwargs)
        def test(self,data,**kwargs):
            examples=self.dataset_loader.load(data)
            if any(e['metadata']['role']!='test' for e in examples):raise ValueError('test role violation')
            before=(copy.deepcopy(self.memory.snapshot()),self.harness_version,copy.deepcopy(self.component_versions),copy.deepcopy(self.task_interface_config.to_dict()),copy.deepcopy(self.router_config.to_dict()),len(self.optimizer.history))
            checkpoint_state=copy.deepcopy(self.checkpoint.state)
            checkpoint_bytes=self.checkpoint.path.read_bytes() if self.checkpoint.path and self.checkpoint.path.exists() else None
            try:return super().test(examples,**kwargs)
            finally:
                after=(self.memory.snapshot(),self.harness_version,self.component_versions,self.task_interface_config.to_dict(),self.router_config.to_dict(),len(self.optimizer.history))
                if before!=after:
                    self.memory.restore(before[0]);self.harness_version=before[1];self.component_versions=before[2]
                    self.task_interface_config=TaskInterfaceConfig(**before[3]);self.router_config=RouterConfig(**before[4])
                    del self.optimizer.history[before[5]:]
                    self.checkpoint.state=checkpoint_state
                    if self.checkpoint.path:
                        if checkpoint_bytes is None:self.checkpoint.path.unlink(missing_ok=True)
                        else:
                            tmp=self.checkpoint.path.with_suffix('.readonly_restore.tmp');tmp.write_bytes(checkpoint_bytes);tmp.replace(self.checkpoint.path)
                    with (root/'readonly_violations.jsonl').open('a') as f:f.write(json.dumps({'restored':True,'reason':'test modified learning state'})+'\n')
                    raise AssertionError('test modified learning state; snapshot restored')
        def continual_train(self,*args,**kwargs):
            if arm=='static_harness':raise PermissionError('Static supports report-only test, not continual learning')
            if not real_execution_authorized:raise PermissionError('真实实验未授权；准备阶段只检查接口')
            if self.forgetting_probe.enabled:raise ValueError('test forgetting probe forbidden')
            return super().continual_train(*args,**kwargs)

    pipeline=MedicalPipeline(model or NoInferenceModel(),evaluator=evaluator,evaluation_recorder=EvaluationRecorder(record_dir=str(root/'evaluator')),optimizer=optimizer,memory=memory,memory_config=config,dataset_loader=StrictMedicalLoader(prepared),forgetting_probe_config={'enabled':False},checkpoint_path=root/'checkpoint.json',resume_from_checkpoint=True,run_train_final=False,task_interface_config=TaskInterfaceConfig(mode='auto',cache_path=str(root/'task_interface/cache.jsonl')),router_config=RouterConfig(use_tool_selector=False,capability_enabled=False))
    return pipeline
