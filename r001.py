"""One isolated arm of the frozen native-coordinate comparison."""
import json,os,sys,time,traceback
from pathlib import Path
R=Path(os.environ['REPRO_ROOT']).resolve();C=Path(__file__).resolve().parent
config_path=Path(os.environ['RUN_CONFIG']);cfg=json.loads(config_path.read_text());out=Path(cfg['storage_dir'])
os.environ.update(HF_HOME=str(R/'cache'),TMPDIR=str(R/'tmp'),TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4')
os.environ.update(HF_HUB_CACHE=str(R/'cache/huggingface/hub'),TORCH_HOME=str(R/'cache/torch'),XDG_CACHE_HOME=str(R/'cache'),CUDA_CACHE_PATH=str(R/'cache/cuda'),TRITON_CACHE_DIR=str(R/'cache/triton'),PIP_CACHE_DIR=str(R/'cache/pip'))
# Java must be provided on PATH by the caller; validate it before reporting caption scores.
sys.path.insert(0,str(C));os.chdir(C)
import torch
from hcl import cli
from hcl.native_geometry import NativePixelModel
from hcl.models import _state_phase
from hcl.pipeline import HCLPipeline
if os.environ.get('FAIL_CLOSED') == '1':
 from hcl.runtime_fail_closed import install_memory_selector_fail_closed
 install_memory_selector_fail_closed(HCLPipeline)
from hcl.totalbaseline import __main__ as raw
class AuditedNative(NativePixelModel):
 scope='online'
 def generate(self,prompt,*,state):
  t=time.monotonic();phase=_state_phase(state);task_id=str(state.get('task_id',''))
  try:return super().generate(prompt,state=state)
  finally:
   category='online_execution'
   if phase=='generate_artifact_candidates':category='candidate_generation'
   elif phase in {'abstract_memory','anchor_selection','skill_evolution'}:category='experience_consolidation'
   elif self.scope.startswith(('memory_gate','validation_current','candidate:')):category='current_validation' if '/validation/' in task_id else 'historical_validation'
   elif self.scope.startswith('test'):category='reporting_test'
   row={'method':cfg['arm'],'scope':self.scope,'category':category,'phase':phase,'task_id':task_id,'seconds':time.monotonic()-t,'image_supplied':bool(self._image_path_from_state(state)),'peak_allocated_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0}
   with (out/'call_costs.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
values={k:v for k,v in cfg['model'].items() if k!='class_path'}
m=AuditedNative(**values)
cli._load_model=lambda *a,**k:m;raw._load_model=lambda *a,**k:m
predict=HCLPipeline._predict_with_router
original_test=HCLPipeline.test
def scoped(self,*args,**kwargs):
 previous=m.scope;m.scope=kwargs.get('progress_label','online')
 try:return predict(self,*args,**kwargs)
 finally:m.scope=previous
HCLPipeline._predict_with_router=scoped
def readonly_test(self,*args,**kwargs):
 before=self.memory.snapshot();versions=self._component_version_snapshot()
 result=original_test(self,*args,**kwargs)
 assert before==self.memory.snapshot(),'Final test changed learning memory'
 assert versions==self._component_version_snapshot(),'Final test changed component versions'
 with (out/'read_only_checks.jsonl').open('a') as f:f.write(json.dumps({'split':kwargs.get('split'),'memory_unchanged':True,'component_versions_unchanged':True})+'\n')
 return result
HCLPipeline.test=readonly_test
started=time.time()
try:
 if cfg['arm']=='raw_zero_shot':
  sys.argv=['baseline','--config',str(config_path)];raw.main()
 else:cli.main(['run','--config',str(config_path),'--splits','test' if cfg['arm']=='static_harness' else 'train,test'])
 (out/'EXIT.json').write_text(json.dumps({'exit_code':0,'started':started,'ended':time.time(),'elapsed_seconds':time.time()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'all_parameters_frozen':all(not p.requires_grad for p in m._model.parameters())},indent=2))
except Exception:
 (out/'FAILURE.txt').write_text(traceback.format_exc());raise
