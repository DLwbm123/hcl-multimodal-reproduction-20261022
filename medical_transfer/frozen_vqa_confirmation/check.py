import os
import sys,json,copy
from pathlib import Path
R=Path(str(Path(os.environ['CAMPAIGN_ROOT'])/'runs/frozen_vqa_confirmation_v1'));sys.path[:0]=[str(R/'code'),os.environ['MEDICAL_SOURCE'],os.environ['HCL_SOURCE']]
from medical_hcl.bridge import construct_pipeline
from runtime_support import install_protocol
from integration import CoordinateBoundary,install_coordinate_boundary
from semantic_boundary import install_semantic_boundary
from template_contract import install_candidate_contract
from task_scoped_memory import install_task_scoped_memory
from frozen_state import restore_frozen
from hcl.models import LocalTransformersModel
P=json.loads((R/'protocol.json').read_text());B=json.loads((R/'budget.json').read_text());state=json.loads((R/'private_frozen_state.json').read_text())
for arm in ['static_no_memory','hcl_scoped']:
 backend=LocalTransformersModel(name='no-load',path=B['model_path'],torch_dtype='bfloat16',temperature=0,local_files_only=True)
 model=CoordinateBoundary(backend);model.scope='CPU';pipe=construct_pipeline(os.environ['PREPARED_DATA'],R/'cpu_check'/arm,model=model,real_execution_authorized=True,arm='static_harness' if arm=='static_no_memory' else 'stability_hcl')
 install_protocol(pipe,model,R/'cpu_check'/arm,P);install_coordinate_boundary(pipe,model)
 if arm=='hcl_scoped':
  install_candidate_contract(pipe);install_semantic_boundary(pipe);install_task_scoped_memory(pipe);restore_frozen(pipe,state)
  # All frozen templates are readable under the unchanged interface.
  for path in state['artifacts']['paths'].values():assert Path(path).is_file()
 assert backend._model is None and pipe.read_only_workers==1 and not pipe.prediction_cache.enabled
es=[json.loads(x) for x in (R/'private_confirmation.jsonl').read_text().splitlines()]
assert len(es)==200 and len({x['metadata']['group_id'] for x in es})==200 and all(x['metadata']['role']=='validation' for x in es)
result={'passed':True,'pipeline_constructions':2,'frozen_memory_and_selector_restored':True,'fresh_caches':True,'confirmation_groups':200,'real_model_calls':0}
(R/'CPU_CHECK.json').write_text(json.dumps(result,indent=2));print(result)
