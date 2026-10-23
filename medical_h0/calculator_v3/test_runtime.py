import unittest,json,tempfile,asyncio
from pathlib import Path
from runtime_state import ExecutionState
from feedback import bounded,pack_request
class RuntimeTests(unittest.TestCase):
 def test_duplicates_idempotent_and_recovery(self):
  s=ExecutionState([]);s.prepare(0);s.call('read_facts',{})
  args={'expression':'a+b','variables':{'a':2,'b':3}}
  self.assertEqual(s.call('calculate',args)[0]['result'],5)
  n=s.tools.calls
  for _ in range(2):self.assertTrue(s.call('calculate',args)[0]['cached'])
  self.assertEqual(s.tools.calls,n);self.assertEqual(len(s.results),1)
  self.assertIn('calculate',[x['function']['name'] for x in s.prepare(4)])
  self.assertFalse(s.call('calculate',{'expression':'1','variables':{}})[1])
  self.assertIsNone(s.tools.answer)
 def test_submission_reserve_and_isolation(self):
  a=ExecutionState([]);b=ExecutionState([]);a.call('read_facts',{});a.prepare(20)
  self.assertIn('execute_calculator',a.allowed());self.assertEqual(b.allowed(),{'read_facts'})
  v={'calculators':[{'name':'synthetic','episode_id':'e','value':5}]}
  self.assertFalse(a.call('submit_answer',v)[1]);self.assertEqual(a.tools.answer,v)
  self.assertIsNone(b.tools.answer)
 def test_no_answer_synthesis(self):
  s=ExecutionState([]);s.prepare(23);self.assertIsNone(s.tools.answer)
  self.assertTrue(s.call('submit_answer',{'calculators':[{'value':5}]})[1]);self.assertIsNone(s.tools.answer)
 def test_nested_parameters_bounded(self):
  x=bounded({'invalid_tool_calls':[{'args':'a'*20000}]*20})
  self.assertLess(len(json.dumps(x)),5000)
 def test_packing_preserves_essential_fields(self):
  from langchain_core.messages import HumanMessage
  ctx={'current_workflow':[],'training':[{'prompt':'original','trajectory':[{'x':'a'*500}]*12,'correct_feedback':[{'name':'x','final_answer':5}]}]*4}
  count=lambda req:len(req[-1].content)
  req,receipt=pack_request(ctx,0,[],'system',count,max_input=2000)
  data=json.loads(req[-1].content);self.assertEqual(len(data['training']),4)
  self.assertTrue(all(t['prompt']=='original' and t['correct_feedback']==[{'name':'x','final_answer':5}] for t in data['training']))
  self.assertGreater(receipt['removed_trajectory_events'],0)
  self.assertEqual(len(ctx['training'][0]['trajectory']),12)
 def test_essential_overflow_fails(self):
  with self.assertRaises(ValueError):pack_request({'training':[{'prompt':'x'*1000,'trajectory':[]}]},0,[],'s',lambda r:len(r[-1].content),max_input=100)
if __name__=='__main__':unittest.main()
