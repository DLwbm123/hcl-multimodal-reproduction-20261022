"""Attach guarded experience guidance without changing the action JSON contract."""
from vqa_graph_bridge import attach_graph
from adapter import SHORT_ANSWER
from episodic import references,install_scope
from adaptive import AdaptiveIndex,install_adaptive_scope,attach_training_evidence,tool_evidence
from pathlib import Path
from contracts import contract_for
from arbitration import strategy
from grounding import use_macro
import json

class Dispatch:
 def __init__(self,graph,boundary,pipe,registry,fixed,index,policy):
  self.graph=graph;self.boundary=boundary;self.pipe=pipe;self.registry=registry or {};self.fixed=fixed;self.index=index;self.policy=policy;self.name='task-bound-agent'
 def generate(self,prompt,*,state):
  key=str(state.get('internal_sample_id',state.get('task_id','')))
  if key not in self.boundary.vqa_ids:return self.boundary.generate(prompt,state=state)
  prompt+='\nImmutable shared output contract: '+SHORT_ANSWER
  if self.fixed:prompt+='\nHistorical examples (different images; illustrate task/answer style only):\n'+json.dumps(self.fixed)+'\n'+SHORT_ANSWER
  q=state.get('question','');mode=strategy(q,getattr(self.graph,'source_arm','incumbent'),getattr(self.graph,'incumbent_policy',{}),getattr(self.graph,'source_policy',{}));independent=mode=='evidence';macro=use_macro(q,getattr(self.graph,'source_arm','incumbent'),getattr(self.graph,'skill_policy',{}));current=self.boundary.backend._image_path_from_state(state)
  refs=self.index.references(self.pipe.memory.raw.records,q,current,key) if self.index else references(self.pipe.memory.raw.records,self.registry,q,key) if self.registry and not self.fixed else []
  evidence=tool_evidence(self.pipe.memory.raw.records,self.registry,q,key) if self.policy and not self.fixed else {}
  with self.graph.trace_path.with_name('private_retrieval_audit.jsonl').open('a') as f:f.write(json.dumps({'query_id':key,'scope':self.boundary.scope,'reference_count':len(refs),'visual':True,'source_ids':[r['source_id'] for r in refs],'tool_evidence_present':bool(evidence),'adaptive':self.index is not None,'independent_selected':independent,'source_strategy':mode,'grounded_macro':macro})+'\n')
  return self.graph.generate(prompt,state={**state,'reference_examples':refs,'reference_images_enabled':True,'tool_evidence':evidence,'evidence_contract':contract_for(q,getattr(self.graph,'task_contracts',{})) if mode!='base' else '','independent_evidence':independent,'grounded_macro':macro,'macro_decision_source':'training_probe' if getattr(self.graph,'source_arm','')=='macro' else 'learned_policy'})

def connect(pipe,boundary,trace,runtime,plan,fixed=None,registry=None,mode='adaptive',policy=True):
 metadata=(pipe.selection_model,pipe.judge_model,pipe.memory_model)
 graph=attach_graph(pipe,boundary,trace,max_tools=plan['max_tools_per_episode'],tool_runtime=runtime,max_action_retries=plan['max_action_retries'])
 index=AdaptiveIndex(boundary.backend.backend,registry,Path(trace).with_name('private_feature_costs.jsonl'),boundary.feature_guard) if registry and not fixed and mode=='adaptive' else None
 if registry and not fixed:
  if index:install_adaptive_scope(pipe,index)
  else:install_scope(pipe,registry)
 pipe.model=Dispatch(graph,boundary,pipe,registry,fixed,index,policy)
 graph.learn_from_batch=lambda examples,episodes,result:attach_training_evidence(pipe,examples,episodes,result)
 assert (pipe.selection_model,pipe.judge_model,pipe.memory_model)==metadata
 return graph
