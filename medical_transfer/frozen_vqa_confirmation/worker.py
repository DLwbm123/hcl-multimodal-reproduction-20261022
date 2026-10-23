import os,sys,json,time,copy,traceback
from pathlib import Path
C=Path(os.environ['CAMPAIGN_ROOT']);R=C/'runs/frozen_vqa_confirmation_v1';role=os.environ['RUN_ROLE'];O=R/role;O.mkdir(exist_ok=True)
B=json.loads((R/'budget.json').read_text());P=json.loads((R/'protocol.json').read_text())
sys.path[:0]=[str(C/'code'),str(R/'code'),os.environ['MEDICAL_SOURCE'],os.environ['HCL_SOURCE']]
from guard import identity
record={**identity(os.getpid()),'mode':'group'};assert record['pgid']==record['pid'];f=C/'registry'/f'{os.getpid()}.tmp';f.write_text(json.dumps(record));f.replace(f.with_suffix('.json'))
os.environ.update(HF_HOME=str(R/'cache'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TMPDIR=str(R/'tmp'),PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',CUDA_CACHE_PATH=str(R/'cache/cuda'),XDG_CACHE_HOME=str(R/'cache'),TORCH_HOME=str(R/'cache/torch'),TRITON_CACHE_DIR=str(R/'cache/triton'))
import tempfile
tempfile.tempdir=str(R/'tmp')
from integration import CoordinateBoundary,install_coordinate_boundary
from semantic_boundary import install_semantic_boundary
from task_scoped_memory import install_task_scoped_memory
from frozen_state import restore_frozen
from template_contract import install_candidate_contract
from runtime_support import install_protocol
from medical_hcl.bridge import construct_pipeline
from medical_hcl.evaluation import MedicalEvaluator,stage_summary
from hcl.models import LocalTransformersModel,_state_phase
import torch

def write(n,x):
 f=O/n;t=f.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(f)
def append(n,x):
 with (O/n).open('a') as f:f.write(json.dumps(x,allow_nan=False)+'\n')
def guard_check():
 status=json.loads((C/'campaign_status.json').read_text())
 if (C/'CAMPAIGN_STOP.json').exists() or time.time()-status['time']>120 or time.time()>=B['early_stop_epoch']:raise RuntimeError('guard unavailable or time exhausted')
class Model(CoordinateBoundary):
 scope='loading';calls=0
 def generate(self,prompt,*,state):
  guard_check();cap=B['max_calls_baselines'] if role=='baselines' else B['max_calls_per_hcl']
  if self.calls>=cap:raise RuntimeError('call cap')
  self.calls+=1;write('model_attempts.json',{'attempts':self.calls});started=time.monotonic();ok=False
  try:answer=super().generate(prompt,state=state);ok=True;return answer
  finally:append('call_costs.jsonl',{'attempt':self.calls,'scope':self.scope,'phase':_state_phase(state),'seconds':time.monotonic()-started,'success':ok,'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
def snapshot(pipe):
 return copy.deepcopy((pipe.memory.snapshot(),pipe.component_versions,pipe.harness_version,pipe.optimizer.history,pipe.router_config.to_dict(),pipe.task_interface_config.to_dict(),pipe.checkpoint.state))
def build(arm):
 pipe=construct_pipeline(os.environ['PREPARED_DATA'],O/'state'/arm,model=model,real_execution_authorized=True,arm='static_harness' if arm=='static_no_memory' else 'stability_hcl')
 install_protocol(pipe,model,O,P);install_coordinate_boundary(pipe,model)
 if arm.startswith('hcl_'):install_candidate_contract(pipe)
 if arm=='hcl_scoped':
  install_semantic_boundary(pipe)
  install_task_scoped_memory(pipe,lambda x:append('task_scope_checks.jsonl',x))
  restore_frozen(pipe,json.loads((R/'private_frozen_state.json').read_text()))
 assert pipe.optimizer.historical_loss_budget==0 and pipe.optimizer.min_format_compliance_rate==1.0
 return pipe

def evaluate(pipe,arm,stage,task):
 es=[e for e in confirmation if e['task_name']==task];before=snapshot(pipe) if pipe else None;preds=[]
 for i,e in enumerate(es):
  guard_check()
  if pipe is None:
   v=model.prepare_example(e);model.scope='confirmation:direct';state={'question':v['question'],'internal_sample_id':e['task_id'],'task_context':{'files':[{'type':'image','path':e['image']}]}}
   pred={'task_id':e['task_id'],'answer':model.generate(v['question'],state=state)}
  else:
   pred=pipe._predict([e],update_memory=False,progress_label='confirmation:'+arm)[0]
   assert snapshot(pipe)==before,'confirmation changed learning state'
  assert all(not p.requires_grad for p in backend._model.parameters())
  preds.append(pred);append('private_predictions.jsonl',{'arm':arm,'stage':stage,'task':task,'prediction':pred});append('read_only_checks.jsonl',{'arm':arm,'stage':stage,'task':task,'index':i,'learning_state_unchanged':True,'parameters_frozen':True})
  write('progress.json',{'phase':'confirmation','arm':arm,'stage':stage,'task':task,'completed':i+1,'calls':model.calls})
 evaluated=evaluator.evaluate(es,preds,split='confirmation:'+arm);assert evaluated['metrics']['total']==P['confirmation_per_task'][task]
 return evaluated['metrics']
try:
 guard_check()
 for _ in range(35):
  if os.getpid() in json.loads((C/'campaign_status.json').read_text())['live_registered']:break
  time.sleep(1)
 else:raise RuntimeError('worker registration not acknowledged')
 write('START.json',{'identity':record,'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'started':time.time(),'budget_started':B['started_epoch']})
 backend=LocalTransformersModel(name='Qwen2.5-VL-7B-Instruct',path=B['model_path'],torch_dtype='bfloat16',attn_implementation='sdpa',max_new_tokens=2048,answer_max_new_tokens=512,temperature=0.,enable_thinking=False,local_files_only=True)
 backend._load();assert not backend._model.training and all(not p.requires_grad for p in backend._model.parameters())
 model=Model(backend,O/'private_coordinate_trace.jsonl');vocab=json.loads((R/'class_vocabulary.json').read_text());evaluator=MedicalEvaluator(vocab)
 confirmation=[json.loads(x) for x in (R/'private_confirmation.jsonl').read_text().splitlines()];assert len(confirmation)==200 and all(e['metadata']['role']=='validation' for e in confirmation)
 results={}
 arm='static_no_memory' if role=='baselines' else 'hcl_scoped'
 pipe=build(arm)
 metrics={'visual_qa':evaluate(pipe,arm,3,'visual_qa')}
 results={'arm':arm,'metrics':metrics,'memory':pipe.memory.summary(),'actual_confirmation_answers':200,'learning_performed':False}
 write('RESULT.json',results);write('EXIT.json',{'exit_code':0,'ended':time.time(),'elapsed_seconds':time.time()-B['started_epoch'],'calls':model.calls,'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
except BaseException:
 (O/'FAILURE.txt').write_text(traceback.format_exc());write('EXIT.json',{'exit_code':1,'ended':time.time()});raise
