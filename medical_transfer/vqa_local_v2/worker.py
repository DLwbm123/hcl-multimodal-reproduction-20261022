"""One neutral-entry worker per arm family, with fixed calls and original clock."""
import os,sys,json,time,copy,traceback
from pathlib import Path
R=Path(os.environ['RUN_ROOT']);role=os.environ['RUN_ROLE'];O=R/role;O.mkdir(exist_ok=True)
sys.path[:0]=[str(R/'code'),str(R/'source'),str(R/'source/medical_transfer/src'),str(R/'source/medical_transfer/development_diagnostic'),str(R/'source/medical_transfer/template_contract'),str(R/'source/medical_transfer/semantic_boundary')]
P=json.loads((R/'protocol.json').read_text());B=json.loads((R/'budget.json').read_text())
os.environ.update(HF_HOME=str(R/'cache'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TMPDIR=str(R/'tmp'),TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',CUDA_CACHE_PATH=str(R/'cache/cuda'),XDG_CACHE_HOME=str(R/'cache'),TORCH_HOME=str(R/'cache/torch'),TRITON_CACHE_DIR=str(R/'cache/triton'),PYTHONDONTWRITEBYTECODE='1')
import tempfile
tempfile.tempdir=str(R/'tmp')
from adapter import install_stream,freeze_anchors,snapshot
from local_update import train_local_batch
from runtime_support import install_protocol
from template_contract import install_candidate_contract
from semantic_boundary import install_semantic_boundary
from medical_hcl.bridge import construct_pipeline
from medical_hcl.evaluation import MedicalEvaluator
from hcl.models import LocalTransformersModel,_state_phase
import torch

def write(name,obj):
 p=O/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2,allow_nan=False));t.replace(p)
def append(name,obj):
 with (O/name).open('a') as f:f.write(json.dumps(obj,allow_nan=False)+'\n')
def guard():
 s=json.loads((R/'resource_status.json').read_text())
 if (R/'HARD_STOP.json').exists() or time.time()-s['time']>120 or time.time()>=B['soft_deadline_epoch']:raise RuntimeError('guard stale or run limit')
class Model(LocalTransformersModel):
 calls=0;scope='loading'
 def generate(self,prompt,*,state):
  guard();cap=P['budget']['max_calls_controls' if role=='controls' else 'max_calls_per_hcl']
  if self.calls>=cap:raise RuntimeError('call cap')
  self.calls+=1;write('model_attempts.json',{'attempts':self.calls});started=time.monotonic();ok=False
  try:answer=super().generate(prompt,state=state);ok=True;return answer
  finally:append('call_costs.jsonl',{'attempt':self.calls,'scope':self.scope,'phase':_state_phase(state),'seconds':time.monotonic()-started,'success':ok,'image_supplied':bool(self._image_path_from_state(state)),'peak_allocated_bytes':torch.cuda.max_memory_allocated()})

def build(arm):
 pipe=construct_pipeline(os.environ['PREPARED_DATA'],O/'state'/arm,model=model,real_execution_authorized=True,arm='static_harness' if arm=='static' else 'stability_hcl')
 install_protocol(pipe,model,O,P)
 # No prompt/artifact search in this frozen experiment.
 install_semantic_boundary(pipe);install_stream(pipe,arm,P)
 assert not pipe.prediction_cache.enabled and not pipe.router_config.final_generation_cache_path
 return pipe

def evaluate(pipe,arm,checkpoint,stage,kind='report'):
 examples=data[stage][kind];before=snapshot(pipe) if pipe else None;preds=[]
 for i,e in enumerate(examples):
  guard();model.scope=f'{kind}:{arm}:{checkpoint}:{stage}'
  if pipe is None:
   state={'question':e['question'],'internal_sample_id':e['task_id'],'task_context':{'files':[{'type':'image','path':e['image']}]}}
   pred={'task_id':e['task_id'],'answer':model.generate(e['question'],state=state)}
  else:pred=pipe._predict([e],update_memory=False,progress_label=model.scope)[0]
  preds.append(pred)
  append('private_predictions.jsonl',{'arm':arm,'checkpoint':checkpoint,'stage':stage,'kind':kind,'prediction':pred})
  write('progress.json',{'phase':kind,'arm':arm,'checkpoint':checkpoint,'stage':stage,'completed':i+1,'total':len(examples),'calls':model.calls})
 if pipe:assert snapshot(pipe)==before,'report changed learning state'
 assert not model._model.training and all(not p.requires_grad for p in model._model.parameters())
 evaluated=evaluator.evaluate(examples,preds,split=f'{kind}:{checkpoint}:{stage}')
 append('read_only_checks.jsonl',{'arm':arm,'checkpoint':checkpoint,'stage':stage,'kind':kind,'count':len(examples),'unchanged':True,'weights_frozen':True})
 append('private_scores.jsonl',{'arm':arm,'checkpoint':checkpoint,'stage':stage,'kind':kind,'metrics':evaluated['metrics'],'rows':evaluated['case_results']})
 return evaluated['metrics']

try:
 guard();write('START.json',{'pid':os.getpid(),'pgid':os.getpgrp(),'started':time.time(),'original_start':B['started_epoch'],'gpu':os.environ['CUDA_VISIBLE_DEVICES']})
 torch.manual_seed(P['seed'])
 model=Model(name='Qwen2.5-VL-7B-Instruct',path=B['model_path'],torch_dtype='bfloat16',attn_implementation='sdpa',max_new_tokens=2048,answer_max_new_tokens=512,temperature=0.,enable_thinking=False,local_files_only=True)
 model._load();assert not model._model.training and all(not p.requires_grad for p in model._model.parameters())
 write('MODEL_READY.json',{'at':time.time(),'parameters_frozen':True,'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
 data=json.loads((R/'private_data.json').read_text());evaluator=MedicalEvaluator(json.loads((Path(os.environ['PREPARED_DATA'])/'class_vocabulary.json').read_text()))
 results={}
 if role=='controls':
  for arm in ['direct','static']:
   pipe=None if arm=='direct' else build(arm)
   metrics={s:evaluate(pipe,arm,'empty',s) for s in P['order']}
   results[arm]={'matrix':{checkpoint:metrics for checkpoint in ['empty']+P['order']},'fixed_cells_reused':True,'actual_report_answers':216};write('partial_results.json',results)
 arms=['fixed_memory'] if role=='controls' else [role]
 for arm in arms:
  pipe=build(arm);matrix={'empty':{s:evaluate(pipe,arm,'empty',s) for s in P['order']}};anchors=[];stages=[]
  for stage in P['order']:
   guard();model.scope=f'learning:{arm}:{stage}'
   write('progress.json',{'phase':'learning','arm':arm,'stage':stage,'calls':model.calls})
   stage_data=data[stage]
   if arm=='fixed_memory':
    for b in range(2):
     es=stage_data['train'][b*24:(b+1)*24]
     pipe._evaluate_train_batch(es,split=f'train:{stage}:{b}')
     append('fixed_updates.jsonl',{'stage':stage,'batch':b,'accepted':True,'policy':'unconditional','memory':pipe.memory.summary()})
   else:
    for b in range(2):
     train_local_batch(pipe,stage_data['train'][b*24:(b+1)*24],stage_data['selection'],anchors,arm,stage,b,lambda row:append('update_decisions.jsonl',row),lambda:guard())
   anchors+=stage_data['anchor'];freeze_anchors(pipe,anchors)
   assert len(pipe.memory.anchor.records)==24*(P['order'].index(stage)+1)
   state={'stage':stage,'memory':pipe.memory.snapshot(),'artifacts':pipe._checkpoint_artifacts()}
   write(f'private_committed_{arm}_{stage}.json',state)
   matrix[stage]={s:evaluate(pipe,arm,stage,s) for s in P['order']}
   anchor_metrics={s:evaluate(pipe,arm,stage,s,kind='anchor') for s in P['order'][:P['order'].index(stage)+1]}
   stages.append({'stage':stage,'anchors':anchor_metrics,'memory':pipe.memory.summary(),'versions':copy.deepcopy(pipe.component_versions)})
   results[arm]={'matrix':matrix,'stages':stages,'optimizer_decisions':pipe.optimizer.history,'fixed_cells_reused':False};write('partial_results.json',results)
  assert pipe.historical_task_probe_sets=={} and pipe.historical_task_reference_predictions=={}
 write('RESULT.json',results);write('EXIT.json',{'exit_code':0,'ended':time.time(),'elapsed_seconds':time.time()-B['started_epoch'],'calls':model.calls,'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
except BaseException:
 (O/'FAILURE.txt').write_text(traceback.format_exc());write('EXIT.json',{'exit_code':1,'ended':time.time()});raise
