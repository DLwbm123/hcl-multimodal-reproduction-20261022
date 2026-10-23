import copy
from types import SimpleNamespace
from task_scoped_memory import install_task_scoped_memory
class Memory:
 def __init__(self):self.records=[]
 def retrieval_candidates(self):
  return [{'memory_id':r['memory_id'],'payload':'unchanged'} for r in self.records]
class Pipe:
 harness_version=global_batch_index=0
 checkpoint=SimpleNamespace(has_state=False)
 read_only_workers=1
 prediction_cache=SimpleNamespace(enabled=False)
 def __init__(self):self.memory=SimpleNamespace(raw=Memory(),abstract=Memory())
 def _predict_one(self,index,example,**kwargs):
  result=[self.memory.raw.retrieval_candidates(),self.memory.abstract.retrieval_candidates()]
  if example.get('fail'):raise RuntimeError('original failure')
  return index,result
p=Pipe();receipts=[];install_task_scoped_memory(p,receipts.append)
records=[{'memory_id':'v','internal_source':{'task_name':'visual_qa'}}, {'memory_id':'c','internal_source':{'task_name':'classification'}}, {'memory_id':'u','internal_source':{}}]
p.memory.raw.records=copy.deepcopy(records);p.memory.abstract.records=copy.deepcopy(records)
before=copy.deepcopy(records)
for i,t,expected in [(9,'visual_qa','v'),(3,'classification','c')]:
 index,result=p._predict_one(i,{'task_name':t,'answer':'unused'})
 assert index==i and all(x==[{'memory_id':expected,'payload':'unchanged'}] for x in result)
assert p.memory.raw.records==p.memory.abstract.records==before
try:p._predict_one(0,{'task_name':'visual_qa','fail':True})
except RuntimeError as e:assert str(e)=='original failure'
else:raise AssertionError('swallowed original exception')
try:p.memory.raw.retrieval_candidates()
except RuntimeError:pass
else:raise AssertionError('task leaked after exception')
try:p._predict_one(0,{'task_name':'unknown'})
except ValueError:pass
else:raise AssertionError('unknown task accepted')
p.memory.raw.records=[]
assert p._predict_one(0,{'task_name':'visual_qa'})[1][0]==[]
p.memory.raw.records=copy.deepcopy(records)
assert p._predict_one(0,{'task_name':'visual_qa'})[1][0][0]['memory_id']=='v'
assert all(x['after']<=x['before'] for x in receipts)
print('7 task-scope checks passed: both pools, ordering, no mutation, exception reset, unknown task, rollback records, receipts')
