"""VQA uses the graph on every router; non-VQA anchors keep their original interface."""
from vqa_graph_bridge import attach_graph
from adapter import SHORT_ANSWER
import json
from episodic import references,install_scope

class Dispatch:
 def __init__(self,graph,boundary,fixed=None,pipe=None,registry=None,visual=True):self.graph=graph;self.boundary=boundary;self.fixed=fixed;self.pipe=pipe;self.registry=registry or {};self.visual=visual;self.name='task-bound-agent'
 def generate(self,prompt,*,state):
  key=str(state.get('internal_sample_id',state.get('task_id','')))
  if key not in self.boundary.vqa_ids:return self.boundary.generate(prompt,state=state)
  prompt+='\nImmutable shared output contract: '+SHORT_ANSWER
  if self.fixed:
   prompt+='\nHistorical examples (different images; illustrate task/answer style only):\n'+json.dumps(self.fixed)+'\n'+SHORT_ANSWER
  refs=references(self.pipe.memory.raw.records,self.registry,state.get('question',''),key) if self.registry else []
  with self.graph.trace_path.with_name('private_retrieval_audit.jsonl').open('a') as f:f.write(json.dumps({'query_id':key,'scope':self.boundary.scope,'reference_count':len(refs),'visual':self.visual,'source_ids':[r['source_id'] for r in refs]})+'\n')
  state={**state,'reference_examples':refs,'reference_images_enabled':self.visual}
  return self.graph.generate(prompt,state=state)

def connect(pipe,boundary,trace,runtime,plan,fixed=None,registry=None,visual=True):
 metadata=(pipe.selection_model,pipe.judge_model,pipe.memory_model)
 graph=attach_graph(pipe,boundary,trace,max_tools=plan['max_tools_per_episode'],tool_runtime=runtime,max_action_retries=plan['max_action_retries'])
 if registry and not fixed:install_scope(pipe,registry)
 pipe.model=Dispatch(graph,boundary,fixed,pipe,registry if not fixed else None,visual)
 assert (pipe.selection_model,pipe.judge_model,pipe.memory_model)==metadata
 return graph
