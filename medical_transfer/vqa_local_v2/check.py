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
for arm in ['static','fixed_memory','plasticity','stability','bulk']:
 out=R/'cpu_check'/arm
 pipe=construct_pipeline(os.environ['PREPARED_DATA'],out,model=backend,real_execution_authorized=True,arm='static_harness' if arm=='static' else 'stability_hcl')
 install_protocol(pipe,backend,out,P)
 # Prompt artifacts stay frozen.
 install_semantic_boundary(pipe);install_stream(pipe,arm,P);pipes[arm]=pipe
 assert not pipe.prediction_cache.enabled and not backend._model and not pipe.memory.config.skill_enabled
 assert pipe.optimizer.components==[]
checks.append('five_fresh_pipelines_no_inference')
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
# Test edit bounds, capacity, preservation and whole-bank matching with synthetic state.
from local_update import candidates,train_local_batch
before={'raw':[{'memory_id':str(i),'question':'old question '+str(i),'internal_source':{'task_id':str(i)}} for i in range(10)],'abstract':[{'abstract_role':r,'memory_id':r,'guidance':'old'} for r in ['main_policy','failure_avoidance']],'anchor':[],'skills':[]}
proposed=copy.deepcopy(before)
proposed['raw']=[{'memory_id':'new'+str(i),'question':'old question '+str(i),'internal_source':{'task_id':'new'+str(i)}} for i in range(10)]
proposed['abstract']=[{**x,'guidance':'new'} for x in proposed['abstract']]
examples=[{'task_id':'new'+str(i)} for i in range(24)]
cs=candidates(before,proposed,examples,0,True);bulk=candidates(before,proposed,examples,0,False)
assert len(cs)==len(bulk)==3
assert cs[0][1]['abstract']==before['abstract'] and cs[1][1]['raw']==before['raw']
assert sum(x in before['raw'] for x in cs[2][1]['raw'])==8
assert cs[2][1]['abstract'][1]==before['abstract'][1]
assert bulk[2][1]==proposed
assert before['raw'][0]['memory_id']=='0'
checks.append('local_max_two_raw_one_abstract_preserves_others_and_bulk_has_same_three_scopes')
# Execute real gate/scoring and transaction logic with a deterministic fake predictor.
from types import MethodType
pipe=pipes['stability'];pipe.memory.restore(before)
current=[];old=[]
for i in range(24):
 e=ex('B','validation');e['task_id']='current'+str(i);e['answer']='1';current.append(e)
 e=ex('A');e['task_id']='old'+str(i);e['answer']='1';old.append(e)
train=[{**ex('B','h_search'),'task_id':'new'+str(i)} for i in range(24)]
def fake_train(self,es,split):self.memory.restore(proposed)
def fake_predict(self,es,**kwargs):
 raw=any(x['memory_id'].startswith('new') for x in self.memory.raw.records)
 abstract=any(x.get('guidance')=='new' for x in self.memory.abstract.records)
 out=[]
 for e in es:
  i=int(re.sub(r'\D','',e['task_id']))
  good=(not raw or i!=0) if e['task_id'].startswith('old') else (abstract and i==0 or raw and i==1)
  out.append({'task_id':e['task_id'],'answer':'1' if good else '2'})
 return out
import re
pipe._evaluate_train_batch=MethodType(fake_train,pipe);pipe._predict=MethodType(fake_predict,pipe)
decisions=[]
train_local_batch(pipe,train,current,old,'stability','B',0,decisions.append,lambda:None)
assert decisions[0]['chosen_index']==1 and pipe.memory.snapshot()==cs[1][1]
assert decisions[0]['candidates'][2]['metrics']['historical_anchor_metrics']['forget_count']==1
pipe.memory.restore(before);pipe.optimizer.historical_loss_budget=None
train_local_batch(pipe,train,current,old,'plasticity','B',0,decisions.append,lambda:None)
assert decisions[-1]['chosen_index']==2
pipe.memory.restore(before)
def fail_predict(self,es,**kwargs):
 if any(x['memory_id'].startswith('new') for x in self.memory.raw.records):raise RuntimeError('synthetic candidate failure')
 return fake_predict(self,es,**kwargs)
pipe._predict=MethodType(fail_predict,pipe)
try:train_local_batch(pipe,train,current,old,'stability','B',0,decisions.append,lambda:None)
except RuntimeError as e:assert 'synthetic' in str(e)
else:raise AssertionError('expected injected failure')
assert pipe.memory.snapshot()==before
checks.append('real_gate_transaction_B0_selects_safe_candidate_Binf_accepts_gain_and_exception_restores')
result={'passed':True,'checks':checks,'model_calls':0,'backend_loaded':backend._model is not None}
(R/'CPU_CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
