import os,sys,json,time,copy,traceback
from pathlib import Path
C=Path(os.environ['CAMPAIGN_ROOT']);R=C/'runs/semantic_boundary_v1'
B=json.loads((R/'budget.json').read_text());P=json.loads((R/'protocol.json').read_text())
sys.path[:0]=[str(C/'code'),str(R/'code'),os.environ['MEDICAL_SOURCE'],os.environ['HCL_SOURCE']]
from guard import identity
record={**identity(os.getpid()),'mode':'group'};assert record['pgid']==record['pid']
f=C/'registry'/f'{os.getpid()}.tmp';f.write_text(json.dumps(record));f.replace(f.with_suffix('.json'))
os.environ.update(HF_HOME=str(R/'cache'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TMPDIR=str(R/'tmp'),PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',CUDA_CACHE_PATH=str(R/'cache/cuda'),XDG_CACHE_HOME=str(R/'cache'),TORCH_HOME=str(R/'cache/torch'),TRITON_CACHE_DIR=str(R/'cache/triton'))
import tempfile
tempfile.tempdir=str(R/'tmp')
from integration import CoordinateBoundary,install_coordinate_boundary
from semantic_boundary import install_semantic_boundary,bounded_context
from runtime_support import install_protocol
from diagnostic_support import memory_pool,lock_learning,learning_state,pair_changes
from medical_hcl.bridge import construct_pipeline
from medical_hcl.evaluation import MedicalEvaluator
from hcl.models import LocalTransformersModel,_state_phase
import torch

def loadrows(name):return [json.loads(x) for x in (R/name).read_text().splitlines()]
def write(name,data):
 f=R/name;t=f.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2,allow_nan=False));t.replace(f)
def append(name,data):
 with (R/name).open('a') as f:f.write(json.dumps(data,allow_nan=False)+'\n')
def guard_check():
 if (C/'CAMPAIGN_STOP.json').exists():raise RuntimeError('campaign stopped')
 status=json.loads((C/'campaign_status.json').read_text())
 if time.time()-status['time']>120:raise RuntimeError('campaign guard stale')
 if time.time()>=B['early_stop_epoch']:raise RuntimeError('experiment deadline')
class Model(CoordinateBoundary):
 scope='loading';condition='none';stage='none';calls=0
 def generate(self,prompt,*,state):
  guard_check()
  if self.calls>=B['max_calls']:raise RuntimeError('call cap')
  self.calls+=1;write('model_attempts.json',{'attempts':self.calls,'started_epoch':B['started_epoch']})
  started=time.monotonic();ok=False
  try:
   answer=super().generate(prompt,state=state);ok=True;return answer
  finally:append('call_costs.jsonl',{'attempt':self.calls,'stage':self.stage,'condition':self.condition,'phase':_state_phase(state),'seconds':time.monotonic()-started,'success':ok,'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
try:
 guard_check()
 for _ in range(30):
  if os.getpid() in json.loads((C/'campaign_status.json').read_text())['live_registered']:break
  time.sleep(1)
 else:raise RuntimeError('campaign did not acknowledge worker before model load')
 backend=LocalTransformersModel(name='Qwen2.5-VL-7B-Instruct',path=B['model_path'],torch_dtype='bfloat16',attn_implementation='sdpa',max_new_tokens=2048,answer_max_new_tokens=512,temperature=0.,enable_thinking=False,local_files_only=True)
 backend._load();assert not backend._model.training and all(not p.requires_grad for p in backend._model.parameters())
 model=Model(backend,R/'private_coordinate_trace.jsonl');snapshot=json.loads((R/'private_frozen_memory.json').read_text());vocab=json.loads((R/'class_vocabulary.json').read_text());evaluator=MedicalEvaluator(vocab)
 all_results={};references=loadrows('private_screen_reference.jsonl');frozen_config=None
 for stage in ['screen','confirmation']:
  model.stage=stage;examples=loadrows('private_'+stage+'.jsonl');assert all(e['metadata']['role']=='validation' for e in examples)
  stage_results={};case_sets={};raw_trace={}
  for condition in P['conditions']:
   model.condition=condition
   if stage=='screen' and condition!='raw_semantic':
    predictions=[x['prediction'] for x in references if x['condition']==condition]
    if condition=='raw_only':raw_trace={p['task_id']:p['trace'] for p in predictions}
   else:
    pipe=construct_pipeline(os.environ['PREPARED_DATA'],R/'state'/stage/condition,model=model,real_execution_authorized=True,arm='stability_hcl')
    install_protocol(pipe,model,R,{'anchors_per_task':24});install_coordinate_boundary(pipe,model)
    if condition=='raw_semantic':install_semantic_boundary(pipe)
    lock_learning(pipe);pipe.memory.restore(memory_pool(snapshot,'static_no_memory' if condition=='static_no_memory' else 'raw_only'))
    before=learning_state(pipe)
    config={name:{k:v for k,v in cfg.to_dict().items() if 'path' not in k} for name,cfg in [('router',pipe.router_config),('interface',pipe.task_interface_config)]}
    if frozen_config is None:frozen_config=copy.deepcopy(config)
    else:assert config==frozen_config
    predictions=[]
    for index,e in enumerate(examples):
     guard_check();prediction=pipe._predict([e],update_memory=False,progress_label=stage+':'+condition)[0]
     assert learning_state(pipe)==before
     trace=prediction.get('trace',{})
     # The only permitted difference is the final context suffix. No retrieval drift.
     if condition=='raw_semantic':
      ref=raw_trace[e['task_id']]
      assert trace['selected_memory']==ref['selected_memory'],'retrieval changed'
      assert trace['workflow_decision']==ref['workflow_decision'],'workflow changed'
      assert trace['prompt']==bounded_context(ref['prompt'],ref['selected_memory']),'unexpected final context change'
     if condition=='raw_only':raw_trace[e['task_id']]=trace
     predictions.append(prediction);append('private_predictions.jsonl',{'stage':stage,'condition':condition,'task':e['task_name'],'prediction':prediction})
     append('read_only_checks.jsonl',{'stage':stage,'condition':condition,'index':index,'learning_state_unchanged':True,'parameters_frozen':all(not p.requires_grad for p in backend._model.parameters()),'retrieval_matched':condition=='raw_semantic'})
     write('progress.json',{'stage':stage,'condition':condition,'completed':index+1,'total':len(examples),'calls':model.calls})
   metrics={};cases={}
   for task in P['tasks']:
    subset=[e for e in examples if e['task_name']==task];ids={e['task_id'] for e in subset};pred=[p for p in predictions if p['task_id'] in ids]
    evaluated=evaluator.evaluate(subset,pred,split=stage+':'+condition+':'+task);assert evaluated['metrics']['total']==(12 if stage=='screen' else 50)
    metrics[task]=evaluated['metrics'];answers={p['task_id']:p['answer'] for p in pred};cases[task]={x['task_id']:{'success':x['success'],'answer':answers[x['task_id']]} for x in evaluated['case_results']}
   stage_results[condition]={'metrics':metrics};case_sets[condition]=cases
   write('partial_results.json',{**all_results,stage:stage_results})
  pairs={c:{t:pair_changes(case_sets['static_no_memory'][t],case_sets[c][t]) for t in P['tasks']} for c in ['raw_only','raw_semantic']}
  m=lambda c,t:stage_results[c]['metrics'][t]
  safety=all(m('raw_semantic',t)['correct']>=m('raw_only',t)['correct'] and m('raw_semantic',t)['format_compliant_count']>=m('raw_only',t)['format_compliant_count'] for t in P['tasks'])
  retain=m('raw_semantic','visual_qa')['correct']>=m('raw_only','visual_qa')['correct']
  reduce=pairs['raw_semantic']['visual_qa']['successful_to_failed']<pairs['raw_only']['visual_qa']['successful_to_failed']
  practical=m('raw_semantic','visual_qa')['primary_score']-m('static_no_memory','visual_qa')['primary_score']>=.05-1e-12
  all_results[stage]={'conditions':stage_results,'paired_changes_vs_static':pairs,'gate':{'retains_raw_vqa_accuracy':retain,'reduces_vqa_losses_of_static_successes':reduce,'no_task_accuracy_or_format_loss_vs_raw':safety,'vqa_gain_at_least_5pp_vs_static':practical,'passed':bool(retain and reduce and safety and (stage=='screen' or practical))}}
  write('partial_results.json',all_results)
 write('RESULT.json',{'stages':all_results,'model_calls':model.calls,'new_answer_slots':486,'reused_screen_answer_slots':72,'parameters_frozen':True,'test_used':False,'policy_count':1,'coordinate_semantic_gate_previously_failed':True})
 write('EXIT.json',{'exit_code':0,'ended':time.time(),'elapsed_seconds':time.time()-B['started_epoch'],'calls':model.calls,'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
except BaseException:
 (R/'FAILURE.txt').write_text(traceback.format_exc());write('EXIT.json',{'exit_code':1,'ended':time.time()});raise
