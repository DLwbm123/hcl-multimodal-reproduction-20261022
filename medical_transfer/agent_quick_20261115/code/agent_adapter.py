"""VQA uses the graph on every router; non-VQA anchors keep their original interface."""
from vqa_graph_bridge import attach_graph
from adapter import SHORT_ANSWER
import json

class Dispatch:
 def __init__(self,graph,boundary,fixed=None):self.graph=graph;self.boundary=boundary;self.fixed=fixed;self.name='task-bound-agent'
 def generate(self,prompt,*,state):
  key=str(state.get('internal_sample_id',state.get('task_id','')))
  if key not in self.boundary.vqa_ids:return self.boundary.generate(prompt,state=state)
  prompt+='\nImmutable shared output contract: '+SHORT_ANSWER
  if self.fixed:
   prompt+='\nHistorical examples (different images; illustrate task/answer style only):\n'+json.dumps(self.fixed)+'\n'+SHORT_ANSWER
  return self.graph.generate(prompt,state=state)

def connect(pipe,boundary,trace,runtime,plan,fixed=None):
 metadata=(pipe.selection_model,pipe.judge_model,pipe.memory_model)
 graph=attach_graph(pipe,boundary,trace,max_tools=plan['max_tools_per_episode'],tool_runtime=runtime,max_action_retries=plan['max_action_retries'])
 pipe.model=Dispatch(graph,boundary,fixed)
 assert (pipe.selection_model,pipe.judge_model,pipe.memory_model)==metadata
 return graph
