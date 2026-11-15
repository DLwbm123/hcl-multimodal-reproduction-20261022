"""Keep original HCL feedback, proposals and gates; change only explicit study boundaries."""
import copy,json
from contextlib import contextmanager
from pathlib import Path
from integration import install_coordinate_boundary
from semantic_boundary import install_semantic_boundary
from task_scoped_memory import install_task_scoped_memory
from template_contract import install_candidate_contract
from runtime_support import install_protocol
from frozen_state import restore_frozen
from medical_hcl.bridge import construct_pipeline

SHORT_ANSWER='Answer the visible question using the current image. Return only the shortest correct answer, without explanation or a full sentence. For a count return digits only. For multiple requested items return a semicolon-separated list. Do not infer findings from historical examples.'

def snapshot(pipe):
 return copy.deepcopy((pipe.memory.snapshot(),pipe.component_versions,pipe.harness_version,pipe.optimizer.history,pipe.router_config.to_dict(),pipe.task_interface_config.to_dict(),pipe.checkpoint.state))

def restore(pipe,state):
 pipe.checkpoint.state={'memory_snapshot':copy.deepcopy(state['memory']),'artifacts':copy.deepcopy(state['artifacts'])}
 pipe.checkpoint.restore_pipeline(pipe)
 assert pipe.memory.snapshot()==state['memory']
 assert pipe.harness_version==state['artifacts']['harness_version']
 assert pipe.component_versions==state['artifacts']['component_versions']

@contextmanager
def protect_all_anchors():
 import hcl.pipeline as module
 old=module._historical_anchors_for_train
 def chosen(anchors,train):
  assert len(anchors)==72
  assert not {x['task_id'] for x in anchors}&{x['task_id'] for x in train}
  return copy.deepcopy(anchors)
 module._historical_anchors_for_train=chosen
 try:yield
 finally:module._historical_anchors_for_train=old

def build(root,out,model,arm,plan,old_state,*,saved=None):
 pipe=construct_pipeline(plan['prepared_data'],out/'state'/arm,model=model,real_execution_authorized=True,arm='static_harness' if arm in ('B','F') else 'stability_hcl')
 install_protocol(pipe,model,out,plan);install_coordinate_boundary(pipe,model)
 if arm not in ('B','F'):
  install_candidate_contract(pipe);install_semantic_boundary(pipe);install_task_scoped_memory(pipe)
  if saved is not None:restore(pipe,saved)
  elif arm in ('resume','H'):restore_frozen(pipe,old_state)
  else:
   pipe.memory.anchor.records=copy.deepcopy(old_state['memory']['anchor'])
   pipe.memory.anchor._persist()
  pipe.optimizer.components=['memory_selector','router_context'];pipe.optimizer.candidate_count=2
  pipe.optimizer.min_correct_gain=1;pipe.optimizer.min_primary_score_delta=1e-9
  pipe.optimizer.historical_loss_budget=plan['gate']['historical_loss_budget']
  pipe.optimizer.min_format_compliance_rate=plan['gate']['min_format_compliance_rate']
  assert len(pipe.memory.historical_anchors())==72
  assert pipe.optimizer.historical_loss_budget==2 and pipe.optimizer.min_format_compliance_rate==.95
 assert pipe.historical_task_probe_sets=={} and not pipe.prediction_cache.enabled and pipe.read_only_workers==1
 return pipe
