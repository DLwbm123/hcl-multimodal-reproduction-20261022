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
 parent=Path(P['parent_run']);assert json.loads((parent/'fresh/EXIT.json').read_text())['exit_code']==0
 backend=LocalTransformersModel(name='Qwen2.5-VL-7B-Instruct',path=P['model_path'],torch_dtype='bfloat16',attn_implementation='sdpa',max_new_tokens=2048,answer_max_new_tokens=512,temperature=0.,enable_thinking=False,local_files_only=True)
 backend._load();model=Model(ReferenceBackend(backend,O/'private_reference_audit.jsonl',P['method']['reference_max_side']),O/'private_coordinate_trace.jsonl');model.feature_guard=guard
 assert not backend._model.training and all(not x.requires_grad for x in backend._model.parameters());write('MODEL_READY.json',{'at':time.time(),'parameters_frozen':True})
 old=json.loads(Path(P['old_state']).read_text());data=json.loads((R/'private_learning.json').read_text());saved=json.loads((R/'private_base_state.json').read_text());registry={str(e['task_id']):{'question':e['question'],'image':e['image']} for e in data['train']}
 evaluator=MedicalEvaluator(json.loads((Path(P['prepared_data'])/'class_vocabulary.json').read_text()));pipe=build(R,O,model,'fresh',P,old,saved=saved);config=copy.deepcopy(P);config['max_tools_per_episode']=1 if role=='fit_one' else 2
 graph=connect(pipe,model,O/'private_graph_traces.jsonl',VisionTools(O/'private_tool_artifacts'),config,registry=registry,mode='adaptive',policy=True)
 if role in ('candidate','confirm_policy'):graph.utility_policy=json.loads((R/'POLICY_CANDIDATE.json').read_text())
 model.arm=role;pipe.model.index.ensure();results={}
 if role in ('fit_full','fit_one'):
  before=snapshot(pipe)
  for e in data['train'][:2]:
   pred=pipe._predict([e],update_memory=False,progress_label='preflight')[0];assert not pred.get('error') and graph.episodes[-1]['status']=='completed'
  assert snapshot(pipe)==before;write('PREFLIGHT.json',{'passed':True,'episodes':2,'max_tools':config['max_tools_per_episode'],'read_only':True});graph.episodes.clear()
  arm='one' if role=='fit_one' else 'full'
  results['train']=evaluate(pipe,arm,data['train'],'counterfactual_train',graph)
  results['selection']=evaluate(pipe,arm,data['selection'],'selection',graph)
  results['historical']=evaluate(pipe,arm,pipe.memory.historical_anchors(),'historical',graph)
 elif role=='candidate':
  results['selection']=evaluate(pipe,'policy',data['selection'],'selection',graph)
  results['historical']=evaluate(pipe,'policy',pipe.memory.historical_anchors(),'historical',graph)
 else:
  assert role in ('confirm_full','confirm_policy') and json.loads((R/'POLICY_GATE.json').read_text())['accepted'];examples=json.loads((R/'private_confirmation.json').read_text());results['confirmation']=evaluate(pipe,role,examples,'confirmation',graph)
 write('RESULT.json',results);write('EXIT.json',{'exit_code':0,'ended':time.time(),'calls':model.calls,'elapsed_seconds':time.time()-B['started_epoch']})
except BaseException:
 (O/'FAILURE.txt').write_text(traceback.format_exc());write('EXIT.json',{'exit_code':1,'ended':time.time()});raise
