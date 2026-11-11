"""Real LangGraph/Message objects with scripted model outputs; not efficacy evidence."""
import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace,MethodType
from vqa_graph_bridge import attach_graph

class Backend:
 scope='cpu_fixture'
 def __init__(self,repeat=False):self.inputs=[];self.repeat=repeat
 def generate(self,prompt,*,state):
  self.inputs.append((prompt,state))
  if state['router_phase']=='graph_tool':return '2'
  if self.repeat or 'Tool observations:\n[]' in prompt:
   return json.dumps({'tool':'answer_visual_question','instruction':'Count visible objects.'})
  return json.dumps({'answer':'2'})

def pipe_fixture():
 metadata=object()
 p=SimpleNamespace(model=object(),selection_model=metadata,judge_model=metadata,memory_model=metadata,read_only_workers=1,prediction_cache=SimpleNamespace(enabled=False),router_config=SimpleNamespace(use_chat_tool_calls=False,final_generation_cache_path=None))
 def batch(self):
  answer=self.model.generate('visible question',state={'question':'How many objects?', 'task_context':{'files':[]},'answer':'DO_NOT_LEAK','task_name':'PRIVATE_TASK'})
  return {'metrics':{'answer':answer}}
 p._evaluate_train_batch=MethodType(batch,p)
 return p

class Checks(unittest.TestCase):
 def test_loop_and_training_feedback(self):
  with tempfile.TemporaryDirectory() as d:
   p=pipe_fixture();metadata=p.selection_model;b=Backend();g=attach_graph(p,b,Path(d)/'trace.jsonl')
   self.assertIs(p.selection_model,metadata);self.assertIs(p.judge_model,metadata);self.assertIs(p.memory_model,metadata)
   result=p._evaluate_train_batch();self.assertEqual(result['metrics']['answer'],'2')
   self.assertEqual(result['metrics']['graph_execution']['tool_counts'],{'answer_visual_question':1})
   self.assertEqual([x['node'] for x in g.episodes[0]['events']],['process','execute','process'])
   self.assertGreater(g.episodes[0]['checkpoint_count'],2)
   for prompt,state in b.inputs:
    self.assertNotIn('DO_NOT_LEAK',prompt);self.assertNotIn('PRIVATE_TASK',prompt);self.assertNotIn('answer',state)
   with self.assertRaisesRegex(ValueError,'already'):attach_graph(p,b,Path(d)/'x')
 def test_rejected_arguments_retry_without_executing_invalid_tool(self):
  class Repair(Backend):
   rejected=False
   def generate(self,prompt,*,state):
    if state['router_phase']=='graph_controller' and 'Tool observations:\n[]' not in prompt and not self.rejected:
     self.rejected=True
     return '{"tool":"answer_visual_question","arguments":{}}'
    return super().generate(prompt,state=state)
  with tempfile.TemporaryDirectory() as d:
   p=pipe_fixture();g=attach_graph(p,Repair(),Path(d)/'trace.jsonl')
   r=p._evaluate_train_batch();self.assertEqual(r['metrics']['answer'],'2')
   self.assertEqual(g.episodes[0]['protocol_errors'],1)
   self.assertEqual(r['metrics']['graph_execution']['tool_counts'],{'answer_visual_question':1})
   self.assertEqual([e['node'] for e in g.episodes[0]['events']],['process','execute','process','validation_error','process'])
 def test_tool_budget_prevents_infinite_loop(self):
  with tempfile.TemporaryDirectory() as d:
   p=pipe_fixture();g=attach_graph(p,Backend(repeat=True),Path(d)/'trace.jsonl',max_tools=2)
   r=p._evaluate_train_batch();self.assertEqual(r['metrics']['answer'],'')
   self.assertEqual(g.episodes[0]['status'],'failed');self.assertEqual(r['metrics']['graph_execution']['tool_counts'],{'answer_visual_question':2})

if __name__=='__main__':unittest.main()
