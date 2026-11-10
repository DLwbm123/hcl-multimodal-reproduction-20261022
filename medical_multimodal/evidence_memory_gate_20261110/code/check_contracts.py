"""Synthetic contract checks: unknown memory, rollback, disjoint gate denominators."""
from tempfile import TemporaryDirectory
from hcl.memory import MemoryConfig
from evidence_memory import EvidenceMemory
from gate import evaluate

with TemporaryDirectory() as d:
    m=EvidenceMemory(MemoryConfig(record_dir=d,abstract_mode='rule',skill_enabled=False,anchor_selection_mode='off'),model=None)
    for i in range(2):
        m.record_training_interaction('group',{'task_id':str(i),'question':'synthetic','answer':{'semantic':None}},
                                       {'task_id':str(i),'answer':{'operation':'recorded'}})
    assert all(x['previous_attempt_correct'] is None and x['correct_feedback']['semantic'] is None for x in m.raw.retrieval_candidates())
    snapshot=m.snapshot();m.clear();m.restore(snapshot);assert m.snapshot()==snapshot

inputs=[{'id':str(i),'gate_role':'current' if i<6 else 'anchor','semantic_feedback_allowed':True} for i in range(8)]
inputs.append({'id':'unknown','gate_role':'current','semantic_feedback_allowed':False})
episodes=[{'id':x['id'],'arm':arm,'status':'completed','predicted':('B' if arm=='memory' and x['id']=='0' else 'A'),
           'tool_calls':0,'seconds':1} for x in inputs for arm in ['Fixed','memory']]
targets={str(i):('B' if i==0 else 'A') for i in range(8)}
report=evaluate(inputs,episodes,targets)
assert report['accepted_for_research_development'] and report['unknown_sources']==1
next(x for x in episodes if x['id']=='6' and x['arm']=='memory')['predicted']='B'
assert not evaluate(inputs,episodes,targets)['accepted_for_research_development']
assert not evaluate(inputs,episodes[:-1],targets)['accepted_for_research_development']
try:evaluate(inputs,episodes,dict(targets,unknown='A'))
except AssertionError:pass
else:raise AssertionError('Unknown semantic target admitted')
print('Unknown feedback, rollback, historical-loss and complete-coverage contracts passed')
