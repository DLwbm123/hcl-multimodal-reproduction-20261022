"""Synthetic gate/identity tests and actual pipeline construction without inference."""
import os,sys,json,copy,tempfile
from pathlib import Path
R=Path(os.environ['RUN_ROOT']);sys.path[:0]=[str(R/'code'),str(R/'source'),str(R/'source/medical_transfer/src'),str(R/'source/medical_transfer/development_diagnostic'),str(R/'source/medical_transfer/template_contract'),str(R/'source/medical_transfer/semantic_boundary')]
from adapter import earlier_stage_anchors,stage_anchor_scope,install_stream,freeze_anchors,fixed_select
from medical_hcl.bridge import construct_pipeline
from runtime_support import install_protocol
from template_contract import install_candidate_contract
from semantic_boundary import install_semantic_boundary
from hcl.task_interface import TaskInterface
from hcl.models import LocalTransformersModel
from hcl.pipeline import HCLPipeline
P=json.loads((R/'protocol.json').read_text());B=json.loads((R/'budget.json').read_text())
checks=[]
from selection import allocate
allocation=allocate({'a':{'x':[{'id':'x'}],'y':[{'id':'y'}]},'b':{'x':[{'id':'x'}]}},{'a':1,'b':1},1)
assert allocation['a'][0]['id']=='y' and allocation['b'][0]['id']=='x'
checks.append('global_allocation_preserves_scarce_groups')
def ex(stage,role='retention_anchor'):
 return {'task_id':stage+role,'task_name':'visual_qa','task_type':'medical_visual_qa','question':'How many objects?','image':'/image.png','metadata':{'stream_stage':stage,'role':role,'answer_kind':'integer'},'answer':'2','split':role}
anchors=[ex('A'),ex('B'),ex('C')]
assert [x['metadata']['stream_stage'] for x in earlier_stage_anchors(anchors,[ex('B','h_search')])]==['A'];checks.append('same_task_earlier_anchors_retained_current_future_excluded')
import hcl.pipeline as module
original=module._historical_anchors_for_train
try:
 with stage_anchor_scope():
  assert module._historical_anchors_for_train is earlier_stage_anchors
  raise ValueError('synthetic')
except ValueError:pass
assert module._historical_anchors_for_train is original;checks.append('anchor_adapter_restored_after_error')
backend=LocalTransformersModel(name='cpu-only',path=B['model_path'],torch_dtype='bfloat16',temperature=0,local_files_only=True);backend.scope='CPU'
pipes={}
for arm in ['static','fixed_memory','plasticity','stability']:
 out=R/'cpu_check'/arm
 pipe=construct_pipeline(os.environ['PREPARED_DATA'],out,model=backend,real_execution_authorized=True,arm='static_harness' if arm=='static' else 'stability_hcl')
 install_protocol(pipe,backend,out,P)
 if arm in ('plasticity','stability'):install_candidate_contract(pipe)
 install_semantic_boundary(pipe);install_stream(pipe,arm,P);pipes[arm]=pipe
 assert not pipe.prediction_cache.enabled and not backend._model and not pipe.memory.config.skill_enabled
 assert pipe.optimizer.components==['memory_selector']
checks.append('four_fresh_pipelines_no_inference')
# Runtime output paths legitimately differ; compare scientific decision settings explicitly.
for key in ['components','candidate_count','selection_objective','commit_policy','min_primary_score_delta','min_correct_gain','min_format_compliance_rate','screening_ratio']:
 assert getattr(pipes['stability'].optimizer,key)==getattr(pipes['plasticity'].optimizer,key)
assert pipes['stability'].optimizer.historical_loss_budget==0 and pipes['plasticity'].optimizer.historical_loss_budget is None
stable={'current_task_metrics':{'primary_score':.25,'correct':6,'format_compliance_rate':1.0}}
candidate={'current_task_metrics':{'primary_score':.3,'correct':7,'format_compliance_rate':1.0},'historical_anchor_metrics':{'forget_count':1,'recovered_count':2}}
assert not pipes['stability']._memory_candidate_decision(stable,candidate)[0]
assert pipes['plasticity']._memory_candidate_decision(stable,candidate)[0]
for pipe in [pipes['stability'],pipes['plasticity']]:
 bad=copy.deepcopy(candidate);bad['current_task_metrics']['correct']=6;bad['current_task_metrics']['primary_score']=.25
 assert not pipe._memory_candidate_decision(stable,bad)[0]
 bad=copy.deepcopy(candidate);bad['current_task_metrics']['format_compliance_rate']=.9
 assert not pipe._memory_candidate_decision(stable,bad)[0]
checks.append('B0_counts_losses_without_recovery_offset_Binf_keeps_gain_and_format_gates')
# Test the independent artifact-selection gate as well as memory acceptance.
artifact={'candidate_id':'one','component':'memory_selector','validation_errors':[]}
cr={'current_task_metrics':candidate['current_task_metrics'],'historical_anchor_metrics':{'total':24,'reference_correct':6,'forget_count':1,'recovered_count':2},'metrics':{'behavior_comparable_count':24,'behavior_change_count':1}}
for name,expected in [('stability',False),('plasticity',True)]:
 chosen=pipes[name].optimizer.select_best_candidate([artifact],{'one':cr},stable)
 assert bool(chosen)==expected,(name,chosen)
checks.append('artifact_gate_B0_vs_Binf')
freeze_anchors(pipes['stability'],anchors[:2]);assert len(pipes['stability'].memory.historical_anchors())==2
checks.append('exact_anchors_not_replaced_by_same_task')
visible=TaskInterface.to_model_visible({'task_id':'secret','task_name':'visual_qa','question':'How many?','metadata':{'stream_stage':'B'},'answer':'secret'})
assert 'stream_stage' not in json.dumps(visible) and 'secret' not in json.dumps(visible)
checks.append('stage_and_answer_not_model_visible')
data=json.loads((R/'private_data.json').read_text());es=[e for s in data.values() for batch in s.values() for e in batch]
assert len(es)==504 and len({e['metadata']['group_id'] for e in es})==504
assert len({e['metadata']['pixel_id'] for e in es})==504
assert {e['task_name'] for e in es}=={'visual_qa'} and all(e['metadata']['role']!='test' for e in es)
checks.append('504_unique_groups_pixels_and_roles_no_test')
# Fixed retrieval can choose across stage origins and has no subtype/stage field input.
mem=pipes['fixed_memory'].memory
mem.raw.records=[{'memory_id':'a','internal_source':{'task_name':'visual_qa','task_id':'past'},'question':'How many objects?','response':'1','feedback':'2','response_is_correct':False,'is_anchor':False}]
selected,trace=fixed_select(mem,current_input={'question':'How many objects?'},workflow_decision={})
assert len(selected)==1 and selected[0]['memory_id']=='a';checks.append('fixed_retrieval_materializes_shared_pool')
result={'passed':True,'checks':checks,'model_calls':0,'backend_loaded':backend._model is not None}
(R/'CPU_CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
