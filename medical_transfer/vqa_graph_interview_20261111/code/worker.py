"""Read-only engineering check of the learned HCL state through the real graph."""
import os,sys,json,time,traceback
from pathlib import Path
R=Path(os.environ['RUN_ROOT']);P=json.loads((R/'protocol.json').read_text());B=json.loads((R/'budget.json').read_text())
sys.path[:0]=[str(R/'code'),str(Path(P['previous_run'])/'code')]+P['python_paths']
os.environ.update(HF_HOME=str(R/'cache'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TMPDIR=str(R/'tmp'),TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',CUDA_CACHE_PATH=str(R/'cache/cuda'),XDG_CACHE_HOME=str(R/'cache'),TORCH_HOME=str(R/'cache/torch'),TRITON_CACHE_DIR=str(R/'cache/triton'),PYTHONDONTWRITEBYTECODE='1')
import tempfile
tempfile.tempdir=str(R/'tmp')
from adapter import build,snapshot,SHORT_ANSWER
from integration import CoordinateBoundary
from vqa_graph_bridge import attach_graph
from vision_tools import VisionTools
from hcl.models import LocalTransformersModel,_state_phase
import torch

def write(n,x):
 p=R/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)
def append(n,x):
 with (R/n).open('a') as f:f.write(json.dumps(x,allow_nan=False)+'\n')
def guard():
 s=json.loads((R/'resource_status.json').read_text())
 if (R/'HARD_STOP.json').exists() or time.time()-s['time']>90 or time.time()>=B['soft_deadline_epoch']:raise RuntimeError('guard stale or deadline')
class Backend(LocalTransformersModel):
 calls=0;scope='linked_graph_inference'
 def generate(self,prompt,*,state):
  guard()
  if self.calls>=P['max_model_calls']:raise RuntimeError('call cap')
  self.calls+=1;write('model_attempts.json',{'attempts':self.calls});start=time.monotonic();ok=False
  try:r=super().generate(prompt,state=state);ok=True;return r
  finally:append('private/call_costs.jsonl',{'attempt':self.calls,'phase':_state_phase(state),'seconds':time.monotonic()-start,'success':ok})
class Boundary(CoordinateBoundary):
 scope='linked_graph_inference'
 def prepare_example(self,e):
  assert e['task_name']=='visual_qa','this opt-in entry is VQA-only'
  v=super().prepare_example(e);v['question']+='\n'+SHORT_ANSWER;return v
try:
 guard();write('START.json',{'pid':os.getpid(),'started_epoch':time.time(),'original_start':B['started_epoch']})
 torch.manual_seed(P['seed'])
 backend=Backend(name='Qwen2.5-VL-7B-Instruct',path=P['model_path'],torch_dtype='bfloat16',attn_implementation='sdpa',max_new_tokens=2048,answer_max_new_tokens=512,temperature=0.,enable_thinking=False,local_files_only=True)
 backend._load();backend.calls=json.loads((R/'attempt_offsets.json').read_text()).get('calls',0) if (R/'attempt_offsets.json').exists() else 0
 assert not backend._model.training and all(not x.requires_grad for x in backend._model.parameters())
 boundary=Boundary(backend,R/'private/coordinates.jsonl');old=json.loads(Path(P['old_state']).read_text());learned=json.loads(Path(P['learned_state']).read_text())
 pipe=build(R,R/'private',boundary,'N',P,old,saved=learned);before=snapshot(pipe)
 metadata=(pipe.selection_model,pipe.judge_model,pipe.memory_model)
 runtime=VisionTools(R/'private/tool_artifacts');graph=attach_graph(pipe,backend,R/'private/graph_traces.jsonl',max_tools=P['max_tools_per_episode'],tool_runtime=runtime,max_action_retries=P['max_action_retries'])
 assert snapshot(pipe)==before and (pipe.selection_model,pipe.judge_model,pipe.memory_model)==metadata
 examples=json.loads((Path(P['previous_run'])/'private_learning.json').read_text())['train'][:P['example_count']]
 write('private/membership.json',{'source':'previously exposed training cases, fixed first three; no holdout access','ids':[e['task_id'] for e in examples]})
 receipts=[]
 for i,e in enumerate(examples):
  n=len(graph.episodes);pred=pipe._predict([e],update_memory=False,progress_label='graph_integration_smoke')[0]
  assert len(graph.episodes)==n+1;episode=graph.episodes[-1];assert snapshot(pipe)==before,'learning state changed during read-only check'
  events=episode['events'];selected=pred.get('trace',{}).get('selected_memory',[])
  receipt={'case_index':i+1,'completed':episode['status']=='completed','event_nodes':[x['node'] for x in events],'tools':[x['tool'] for x in events if x['node']=='execute'],'controller_decisions_model_generated':all(x.get('decision_source')=='model' for x in events if x['node']=='process'),'controller_calls':episode.get('controller_calls'),'tool_calls':episode.get('tool_calls'),'graph_checkpoints':episode.get('checkpoint_count'),'selected_memory_count':len(selected),'read_only_state_equal':True,'protocol_errors':episode.get('protocol_errors',0),'failure_reason':episode.get('error')}
  receipts.append(receipt);append('private/predictions.jsonl',pred);write('progress.json',{'completed':i+1,'total':len(examples),'calls':backend.calls});write('partial_receipts.json',receipts)
 result={'purpose':'engineering integration, not accuracy evaluation','cases':receipts,'all_graphs_completed':all(x['completed'] for x in receipts),'no_prescribed_tool_plan':graph.coverage_plan is None,'model_calls':backend.calls,'memory':pipe.memory.summary(),'learned_state_preserved':snapshot(pipe)==before,'metadata_backends_preserved':(pipe.selection_model,pipe.judge_model,pipe.memory_model)==metadata,'parameters_frozen':all(not x.requires_grad for x in backend._model.parameters()),'new_learning_run':False,'previous_confirmation_not_reused':True,'peak_gpu_bytes':torch.cuda.max_memory_allocated()}
 write('RESULT.json',result);write('EXIT.json',{'exit_code':0,'ended':time.time(),'elapsed_seconds':time.time()-B['started_epoch']})
except BaseException:
 (R/'FAILURE.txt').write_text(traceback.format_exc());write('EXIT.json',{'exit_code':1,'ended':time.time()});raise
