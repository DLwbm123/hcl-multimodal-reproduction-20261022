import os,sys,json,time,copy,traceback
from pathlib import Path
R=Path(os.environ['RUN_ROOT']);role=os.environ['RUN_ROLE'];O=R/role;O.mkdir(exist_ok=True)
P=json.loads((R/'protocol.json').read_text());B=json.loads((R/'budget.json').read_text())
sys.path[:0]=[str(R/'code')]+P['python_paths']
os.environ.update(HF_HOME=str(R/'cache'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TMPDIR=str(R/'tmp'),TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',CUDA_CACHE_PATH=str(R/'cache/cuda'),XDG_CACHE_HOME=str(R/'cache'),TORCH_HOME=str(R/'cache/torch'),TRITON_CACHE_DIR=str(R/'cache/triton'),PYTHONDONTWRITEBYTECODE='1')
import tempfile
tempfile.tempdir=str(R/'tmp')
from adapter import build,snapshot,protect_all_anchors,SHORT_ANSWER
from integration import CoordinateBoundary
from hcl.models import LocalTransformersModel,_state_phase
from medical_hcl.evaluation import MedicalEvaluator
import torch

def write(n,x):
 p=O/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)
def append(n,x):
 with (O/n).open('a') as f:f.write(json.dumps(x,allow_nan=False)+'\n')
def guard():
 s=json.loads((R/'resource_status.json').read_text())
 if (R/'HARD_STOP.json').exists() or time.time()-s['time']>120 or time.time()>=B['soft_deadline_epoch']:raise RuntimeError('guard stale or original deadline')
class Model(CoordinateBoundary):
 calls=0;scope='loading';arm='';fixed=[];vqa_ids=set()
 def prepare_example(self,e):
  v=super().prepare_example(e)
  if e['task_name']=='visual_qa':
   self.vqa_ids.add(str(e['task_id']));v['question']=v['question']+'\n'+SHORT_ANSWER
  return v
 def generate(self,prompt,*,state):
  guard();cap=P['budget']['max_calls_per_worker']
  if self.calls>=cap:raise RuntimeError('call cap')
  self.calls+=1;write('model_attempts.json',{'attempts':self.calls});start=time.monotonic();ok=False
  phase=_state_phase(state)
  if phase=='final_answer' and str(state.get('internal_sample_id',state.get('task_id',''))) in self.vqa_ids:prompt+='\nImmutable shared output contract: '+SHORT_ANSWER
  # The fixed-example control receives the same six predefined training examples.
  if self.arm=='F' and phase=='final_answer':
   prompt+='\nHistorical examples (different images; illustrate task/answer style only):\n'+json.dumps(self.fixed)+'\n'+SHORT_ANSWER
  try:answer=super().generate(prompt,state=state);ok=True;return answer
  finally:append('call_costs.jsonl',{'attempt':self.calls,'arm':self.arm,'scope':self.scope,'phase':phase,'seconds':time.monotonic()-start,'success':ok,'peak_allocated_bytes':torch.cuda.max_memory_allocated()})

def evaluate(pipe,arm,examples,split):
 before=snapshot(pipe);preds=[];model.arm=arm
 for i,e in enumerate(examples):
  guard();model.scope=f'{split}:{arm}'
  pred=pipe._predict([e],update_memory=False,progress_label=model.scope)[0];preds.append(pred)
  append('private_predictions.jsonl',{'arm':arm,'split':split,'prediction':pred})
  write('progress.json',{'phase':split,'arm':arm,'completed':i+1,'total':len(examples),'calls':model.calls})
 assert snapshot(pipe)==before,'evaluation altered learning state'
 assert not backend._model.training and all(not x.requires_grad for x in backend._model.parameters())
 result=evaluator.evaluate(examples,preds,split=split)
 append('private_scores.jsonl',{'arm':arm,'split':split,**result})
 append('read_only_checks.jsonl',{'arm':arm,'split':split,'count':len(examples),'unchanged':True,'weights_frozen':True})
 return result['metrics']
try:
 guard();assert not (O/'START.json').exists();write('START.json',{'pid':os.getpid(),'pgid':os.getpgrp(),'started':time.time(),'original_start':B['started_epoch']})
 torch.manual_seed(P['seed'])
 backend=LocalTransformersModel(name='Qwen2.5-VL-7B-Instruct',path=B['model_path'],torch_dtype='bfloat16',attn_implementation='sdpa',max_new_tokens=2048,answer_max_new_tokens=512,temperature=0.,enable_thinking=False,local_files_only=True)
 backend._load();model=Model(backend,O/'private_coordinate_trace.jsonl')
 offsets=json.loads((R/'attempt_call_offsets.json').read_text()) if (R/'attempt_call_offsets.json').exists() else {}
 model.calls=int(offsets.get(role,0))
 assert not backend._model.training and all(not x.requires_grad for x in backend._model.parameters())
 write('MODEL_READY.json',{'at':time.time(),'parameters_frozen':True})
 old=json.loads(Path(P['old_state']).read_text());data=json.loads((R/'private_learning.json').read_text());model.fixed=data['fixed_examples']
 evaluator=MedicalEvaluator(json.loads((Path(P['prepared_data'])/'class_vocabulary.json').read_text()));results={}
 if role in ('fresh','resume'):
  pipe=build(R,O,model,role,P,old);model.arm=role
  preflight=[data['train'][0],next(e for e in pipe.memory.historical_anchors() if e['task_name']=='polyp_localization')]
  for e in preflight:
   pred=pipe._predict([e],update_memory=False,progress_label='preflight')[0]
   assert not pred.get('error') and pred.get('status','ok')=='ok',pred.get('error')
  write('PREFLIGHT.json',{'passed':True,'cases':2,'accuracy_not_used_for_selection':True})
  write('private_initial_state.json',{'memory':pipe.memory.snapshot(),'artifacts':pipe._checkpoint_artifacts()})
  for batch in range(2):
   es=data['train'][24*batch:24*(batch+1)];before=copy.deepcopy(pipe.memory.snapshot());version=pipe.harness_version
   write('progress.json',{'phase':'learning','batch':batch+1,'calls':model.calls})
   with protect_all_anchors():
    result=pipe.train(es,val_data=data['selection'],split=f'new_vqa_batch_{batch+1}',batchsize=24)
   saved={'stage':batch+1,'memory':pipe.memory.snapshot(),'artifacts':pipe._checkpoint_artifacts()}
   write(f'private_committed_batch{batch+1}.json',saved);write('private_final_state.json',saved)
   append('learning_receipts.jsonl',{'batch':batch+1,'before_version':version,'after_version':pipe.harness_version,'memory_changed':before!=pipe.memory.snapshot(),'memory_summary':pipe.memory.summary(),'candidate_count':len(result.generated_candidates),'decisions':copy.deepcopy(pipe.optimizer.history)})
  results[role]=evaluate(pipe,role,data['development'],'development')
 elif role=='controls':
  for arm in ['B','F','H']:
   reuse=R/'reused_control_B.json'
   if arm=='B' and reuse.exists():
    prior=json.loads(reuse.read_text());assert prior['metrics']['total']==80
    results[arm]=prior['metrics'];write('REUSED_CONTROL.json',prior);write('partial_results.json',results);continue
   pipe=build(R,O,model,arm,P,old);results[arm]=evaluate(pipe,arm,data['development'],'development');write('partial_results.json',results)
 else:
  # Only this phase opens the held-out labels, after arm/control selection is immutable.
  frozen=json.loads((R/'FROZEN_SELECTION.json').read_text());assert frozen['locked_before_confirmation']
  examples=json.loads((R/'private_confirmation.json').read_text());arm=role.removeprefix('confirm_')
  saved=json.loads((R/frozen['selected_arm']/'private_final_state.json').read_text()) if arm=='N' else None
  pipe=build(R,O,model,arm,P,old,saved=saved);results[arm]=evaluate(pipe,arm,examples,'confirmation')
 write('RESULT.json',results);write('EXIT.json',{'exit_code':0,'ended':time.time(),'calls':model.calls,'elapsed_seconds':time.time()-B['started_epoch']})
except BaseException:
 (O/'FAILURE.txt').write_text(traceback.format_exc());write('EXIT.json',{'exit_code':1,'ended':time.time()});raise
