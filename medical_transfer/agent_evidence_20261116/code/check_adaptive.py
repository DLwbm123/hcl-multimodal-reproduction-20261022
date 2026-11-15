import unittest,copy,json
from adaptive import eligible,tool_evidence,attach_training_evidence
from types import SimpleNamespace

class Checks(unittest.TestCase):
 def test_exact_and_semantic_bounds(self):
  self.assertTrue(eligible('How many instruments?','HOW MANY INSTRUMENTS?\ncontract',.1,.9))
  self.assertTrue(eligible('What is the number of instruments?','How many instruments?',.95,.9))
  self.assertFalse(eligible('What is the number of instruments?','How many instruments?',.85,.9))
 def test_intent_and_object_mismatch(self):
  self.assertFalse(eligible('How many polyps?','How many instruments?',1,.8))
  self.assertFalse(eligible('Where is the instrument?','How many instruments?',1,.8))
  self.assertFalse(eligible('Describe the scene','What is visible?',1,.8))
 def test_policy_has_no_answers_or_freeform_instructions(self):
  records=[{'internal_source':{'task_id':s},'question':'SECRET_QUESTION','feedback':'SECRET_ANSWER','execution_experience':{'intent':'count','tools':['answer_visual_question'],'first_correct':True,'final_correct':True}} for s in ['a','b']]
  registry={'a':{},'b':{}};before=copy.deepcopy(records);x=tool_evidence(records,registry,'How many instruments?','query')
  self.assertEqual(x['patterns'][0]['episodes'],2);self.assertNotIn('SECRET',json.dumps(x));self.assertEqual(records,before)
  self.assertEqual(tool_evidence(records,registry,'How many instruments?','a'),{})
  self.assertEqual(tool_evidence(records,{'a':{}},'How many instruments?','query'),{})
 def test_roll_back_removes_guidance(self):
  records=[{'internal_source':{'task_id':s}} for s in ['a','b']];original=copy.deepcopy(records)
  for r in records:r['execution_experience']={'intent':'count','tools':['answer_visual_question'],'first_correct':False,'final_correct':True}
  self.assertTrue(tool_evidence(records,{'a':{},'b':{}},'How many?','query'))
  records=original;self.assertEqual(tool_evidence(records,{'a':{},'b':{}},'How many?','query'),{})
 def test_training_feedback_boundary(self):
  record={'internal_source':{'task_id':'a'}};raw=SimpleNamespace(records=[record],_persist=lambda:None);pipe=SimpleNamespace(memory=SimpleNamespace(raw=raw))
  example={'task_id':'a','task_name':'visual_qa','split':'train','question':'How many instruments?','answer':'2'}
  ep={'sample_id':'a','events':[{'node':'execute','tool':'answer_visual_question','result':'2'}]}
  attach_training_evidence(pipe,[example],[ep],{'case_results':[{'task_id':'a','success':True}]})
  self.assertTrue(record['execution_experience']['first_correct']);self.assertNotIn('answer',record['execution_experience'])
  with self.assertRaises(AssertionError):attach_training_evidence(pipe,[{**example,'split':'confirmation'}],[ep],{'case_results':[]})
 def test_invalid_tool_not_exposed(self):
  records=[{'internal_source':{'task_id':s},'execution_experience':{'intent':'count','tools':['shell'],'first_correct':True,'final_correct':True}} for s in ['a','b']]
  self.assertEqual(tool_evidence(records,{'a':{},'b':{}},'How many?','query'),{})
if __name__=='__main__':unittest.main()
