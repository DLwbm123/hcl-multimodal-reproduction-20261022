import os,sys,json,time,copy,traceback
from pathlib import Path
R=Path(os.environ['RUN_ROOT']);role=os.environ['RUN_ROLE'];O=R/role;O.mkdir(exist_ok=True)
P=json.loads((R/'protocol.json').read_text());B=json.loads((R/'budget.json').read_text());sys.path[:0]=[str(R/'code')]+P['python_paths']
os.environ.update(HF_HOME=str(R/'cache'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TMPDIR=str(R/'tmp'),TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',CUDA_CACHE_PATH=str(R/'cache/cuda'),XDG_CACHE_HOME=str(R/'cache'),TORCH_HOME=str(R/'cache/torch'),TRITON_CACHE_DIR=str(R/'cache/triton'),PYTHONDONTWRITEBYTECODE='1')
import tempfile
tempfile.tempdir=str(R/'tmp')
from adapter import build,snapshot,protect_all_anchors,SHORT_ANSWER
from integration import CoordinateBoundary
from agent_adapter import connect
from vision_tools import VisionTools
from episodic import ReferenceBackend
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
 calls=0;scope='loading';arm='';vqa_ids=None
 def __init__(self,*a,**k):super().__init__(*a,**k);self.vqa_ids=set()
 def prepare_example(self,e):
  v=super().prepare_example(e)
  if e['task_name']=='visual_qa':self.vqa_ids.add(str(e['task_id']));v['question']+='\n'+SHORT_ANSWER
  return v
 def generate(self,prompt,*,state):
  guard()
  if self.calls>=P['budget']['max_calls_per_worker']:raise RuntimeError('call cap')
  self.calls+=1;write('model_attempts.json',{'attempts':self.calls});start=time.monotonic();ok=False;phase=_state_phase(state)
  if phase=='final_answer' and str(state.get('internal_sample_id',state.get('task_id',''))) in self.vqa_ids:prompt+='\nImmutable shared output contract: '+SHORT_ANSWER
  original_system=self.config.system_prompt
  if phase=='graph_controller':self.config.system_prompt='You select a tool action. Output exactly one JSON object matching the provided action contract. Never answer the medical question in this role.'
  elif phase=='graph_finalize':self.config.system_prompt='You answer medical visual questions concisely in the requested answer format. Do not output tool calls or JSON action wrappers.'
  try:answer=super().generate(prompt,state=state);ok=True;return answer
  finally:
   self.config.system_prompt=original_system
   append('call_costs.jsonl',{'attempt':self.calls,'arm':self.arm,'scope':self.scope,'phase':phase,'seconds':time.monotonic()-start,'success':ok,'peak_allocated_bytes':torch.cuda.max_memory_allocated()})

def summary(graph):
 if graph is None:return {'episodes':0}
 es=graph.episodes;counts={};errors=0
 for e in es:
  errors+=sum(x['node']=='validation_error' for x in e['events'])
  for x in e['events']:
   if x['node']=='execute':counts[x['tool']]=counts.get(x['tool'],0)+1
 return {'episodes':len(es),'completed':sum(e['status']=='completed' for e in es),'failed':sum(e['status']!='completed' for e in es),'tool_counts':counts,'validation_errors':errors,'controller_calls':sum(x['node']=='process' for e in es for x in e['events'])}
def evaluate(pipe,arm,examples,split,graph):
 before=snapshot(pipe);preds=[];model.arm=arm
 for i,e in enumerate(examples):
  guard();model.scope=f'{split}:{arm}';n=len(graph.episodes) if graph else 0
  pred=pipe._predict([e],update_memory=False,progress_label=model.scope)[0];preds.append(pred)
  if graph and e['task_name']=='visual_qa':assert len(graph.episodes)==n+1,'VQA bypassed graph'
  append('private_predictions.jsonl',{'arm':arm,'split':split,'prediction':pred})
  write('progress.json',{'phase':split,'arm':arm,'completed':i+1,'total':len(examples),'calls':model.calls})
 assert snapshot(pipe)==before,'evaluation altered learning state'
 assert not backend._model.training and all(not x.requires_grad for x in backend._model.parameters())
 result=evaluator.evaluate(examples,preds,split=split);append('private_scores.jsonl',{'arm':arm,'split':split,**result})
 append('read_only_checks.jsonl',{'arm':arm,'split':split,'count':len(examples),'unchanged':True,'weights_frozen':True});write('GRAPH_SUMMARY.json',summary(graph))
 return result['metrics']
try:
 guard();assert not (O/'START.json').exists();write('START.json',{'pid':os.getpid(),'pgid':os.getpgrp(),'started':time.time(),'original_start':B['started_epoch']})
 torch.manual_seed(P['seed'])
 backend=LocalTransformersModel(name='Qwen2.5-VL-7B-Instruct',path=P['model_path'],torch_dtype='bfloat16',attn_implementation='sdpa',max_new_tokens=2048,answer_max_new_tokens=512,temperature=0.,enable_thinking=False,local_files_only=True)
 backend._load();model=Model(ReferenceBackend(backend,O/'private_reference_audit.jsonl',P['method']['reference_max_side']),O/'private_coordinate_trace.jsonl');assert not backend._model.training and all(not x.requires_grad for x in backend._model.parameters())
 model.feature_guard=guard
 offsets=json.loads((R/'attempt_call_offsets.json').read_text()) if (R/'attempt_call_offsets.json').exists() else {}
 model.calls=int(offsets.get(role,0))
 write('MODEL_READY.json',{'at':time.time(),'parameters_frozen':True})
 old=json.loads(Path(P['old_state']).read_text());data=json.loads((R/'private_learning.json').read_text());evaluator=MedicalEvaluator(json.loads((Path(P['prepared_data'])/'class_vocabulary.json').read_text()));results={};graph=None
 registry={str(e['task_id']):{'question':e['question'],'image':e['image']} for e in data['train']}
 if role in ('fresh','reference'):
  initial=None
  pipe=build(R,O,model,role,P,old,saved=initial);model.arm='GN' if role=='fresh' else 'GR';graph=connect(pipe,model,O/'private_graph_traces.jsonl',VisionTools(O/'private_tool_artifacts'),P,registry=registry,mode='adaptive' if role=='fresh' else 'exact',policy=role=='fresh')
  if pipe.model.index is not None:pipe.model.index.ensure()
  before=snapshot(pipe);preflight=[]
  for e in data['train'][:6]+[next(e for e in pipe.memory.historical_anchors() if e['task_name']=='polyp_localization'),next(e for e in pipe.memory.historical_anchors() if e['task_name']=='classification')]:
   n=len(graph.episodes);pred=pipe._predict([e],update_memory=False,progress_label='preflight')[0]
   assert not pred.get('error') and pred.get('status','ok')=='ok',pred.get('error')
   if e['task_name']=='visual_qa':assert len(graph.episodes)==n+1 and graph.episodes[-1]['status']=='completed'
   else:assert len(graph.episodes)==n
   preflight.append({'task':e['task_name'],'executed':True,'agent':e['task_name']=='visual_qa'})
  assert snapshot(pipe)==before
  write('PREFLIGHT.json',{'passed':True,'cases':preflight,'accuracy_not_used_for_selection':True,'graph':summary(graph)})
  # Exercise real multi-image and text-ablation inputs before the guarded learning path.
  target=data['train'][0];refs=[{'source_id':e['task_id'],'path':e['image'],'question':e['question'],'answer':e['answer']} for e in data['train'][1:3]]
  model.scope='reference_preflight';before=snapshot(pipe)
  for visual in [True,False]:
   answer=model.generate('Answer only the current target question: '+target['question'],state={'internal_sample_id':target['task_id'],'question':target['question'],'task_context':{'files':[{'type':'image','path':target['image']}]},'reference_examples':refs,'reference_images_enabled':visual,'router_phase':'graph_finalize','max_new_tokens':64})
   assert answer,'empty reference preflight'
  assert snapshot(pipe)==before
  write('REFERENCE_PREFLIGHT.json',{'passed':True,'visual_and_text_exercised':True,'accuracy_not_used':True,'weights_frozen':all(not x.requires_grad for x in backend._model.parameters())})
  write('private_initial_state.json',{'memory':pipe.memory.snapshot(),'artifacts':pipe._checkpoint_artifacts()})
  evaluate(pipe,'G0',pipe.memory.historical_anchors(),'historical_initial',graph)
  model.arm='GN' if role=='fresh' else 'GR'
  for batch in range(P['learning']['batches']):
   es=data['train'][24*batch:24*(batch+1)];before=copy.deepcopy(pipe.memory.snapshot());version=pipe.harness_version
   write('progress.json',{'phase':'learning','batch':batch+1,'calls':model.calls})
   with protect_all_anchors():result=pipe.train(es,val_data=data['selection'],split=f'agent_vqa_batch_{batch+1}',batchsize=24)
   saved={'stage':batch+1,'memory':pipe.memory.snapshot(),'artifacts':pipe._checkpoint_artifacts()};write(f'private_committed_batch{batch+1}.json',saved);write('private_final_state.json',saved)
   append('learning_receipts.jsonl',{'batch':batch+1,'before_version':version,'after_version':pipe.harness_version,'memory_changed':before!=pipe.memory.snapshot(),'memory_summary':pipe.memory.summary(),'candidate_count':len(result.generated_candidates),'decisions':copy.deepcopy(pipe.optimizer.history),'graph_training':result.evaluation['metrics'].get('graph_execution')})
   write('GRAPH_SUMMARY.json',summary(graph))
  results[model.arm]=evaluate(pipe,model.arm,data['development'],'development',graph)
  evaluate(pipe,model.arm,pipe.memory.historical_anchors(),'historical_final',graph)
 else:
  frozen=json.loads((R/'FROZEN_SELECTION.json').read_text());assert frozen['locked_before_confirmation']
  arm=role.removeprefix('confirm_');examples=json.loads((R/'private_confirmation.json').read_text());saved=None
  if arm in ('GN','GP','GE','GR'):saved=json.loads((R/('reference' if arm=='GR' else 'fresh')/'private_final_state.json').read_text())
  if arm in ['GT','DT']:saved=json.loads(Path(P['learned_direct_state']).read_text())
  build_arm='B' if arm in ['GF','D0'] else 'fresh' if arm=='G0' else 'N'
  pipe=build(R,O,model,build_arm,P,old,saved=saved)
  if arm.startswith('G'):graph=connect(pipe,model,O/'private_graph_traces.jsonl',VisionTools(O/'private_tool_artifacts'),P,data['fixed_examples'] if arm=='GF' else None,registry=registry,mode='exact' if arm in ('GR','GE','GF') else 'adaptive',policy=arm in ('GN','GE'))
  results[arm]=evaluate(pipe,arm,examples,'confirmation',graph)
 write('RESULT.json',results);write('EXIT.json',{'exit_code':0,'ended':time.time(),'calls':model.calls,'elapsed_seconds':time.time()-B['started_epoch']})
except BaseException:
 (O/'FAILURE.txt').write_text(traceback.format_exc());write('EXIT.json',{'exit_code':1,'ended':time.time()});raise
