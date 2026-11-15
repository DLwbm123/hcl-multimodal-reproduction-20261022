import unittest,copy
from types import SimpleNamespace
from episodic import references,install_scope

def row(sid,answer='yes',q='Is a polyp present?'):
 return {'memory_id':sid,'internal_source':{'task_name':'visual_qa','task_id':sid},'question':q+'\nContract','feedback':answer,'response':'WRONG'}
class Checks(unittest.TestCase):
 def setUp(self):
  self.registry={k:{'question':'Is a polyp present?','image':k+'.jpg'} for k in ['a','b','c']}
 def test_only_current_memory_and_training_registry(self):
  self.assertEqual(references([],self.registry,'Is a polyp present?','query'),[])
  self.assertEqual(references([row('selection')],self.registry,'Is a polyp present?','query'),[])
 def test_query_self_and_wrong_question_excluded(self):
  self.assertEqual(references([row('a')],self.registry,'Is a polyp present?','a'),[])
  self.assertEqual(references([row('a')],self.registry,'What color is it?','query'),[])
 def test_corrected_feedback_and_diverse_answers(self):
  rows=[row('a'),row('b'),row('c','no')];before=copy.deepcopy(rows)
  result=references(rows,self.registry,'IS A POLYP PRESENT?\nContract','query')
  self.assertEqual([x['source_id'] for x in result],['a','c']);self.assertEqual(rows,before)
  self.assertNotIn('WRONG',str(result));self.assertTrue(all('feedback' not in x for x in self.registry.values()))
 def test_scope_sanitizes_only_vqa_and_restores(self):
  records=[row('a'),row('b',q='What color?')]
  raw=SimpleNamespace(records=records,retrieval_candidates=lambda:[{'memory_id':r['memory_id'],'previous_attempt':'WRONG','correct_feedback':r['feedback']} for r in records])
  p=SimpleNamespace(memory=SimpleNamespace(raw=raw));p._predict_one=lambda pos,e,**kw:p.memory.raw.retrieval_candidates()
  install_scope(p,self.registry)
  out=p._predict_one(0,{'task_name':'visual_qa','question':'Is a polyp present?'})
  self.assertEqual(len(out),1);self.assertNotIn('previous_attempt',out[0]);self.assertEqual(len(p._predict_one(0,{'task_name':'classification','question':'Classify'})),2)
if __name__=='__main__':unittest.main()
