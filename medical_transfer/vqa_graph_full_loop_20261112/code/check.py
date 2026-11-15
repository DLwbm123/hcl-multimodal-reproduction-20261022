import os,sys,json,copy,tempfile
from pathlib import Path
R=Path(os.environ['RUN_ROOT']);P=json.loads((R/'protocol.json').read_text());sys.path[:0]=[str(R/'code')]+P['python_paths']
from adapter import build,snapshot,protect_all_anchors,SHORT_ANSWER
from integration import CoordinateBoundary
from hcl.models import LocalTransformersModel
from medical_hcl.evaluation import validate_gold
old=json.loads(Path(P['old_state']).read_text());data=json.loads((R/'private_learning.json').read_text());vocab=json.loads((Path(P['prepared_data'])/'class_vocabulary.json').read_text())
# Do not open confirmation labels. Integrity uses the already frozen membership only.
member=json.loads((R/'private_membership.json').read_text())['members'];flat=[x for v in member.values() for x in v]
assert len(flat)==376==len({x['group_id'] for x in flat})==len({x['pixel_id'] for x in flat})
for role in ['train','selection','development']:
 for e in data[role]:validate_gold(e,vocab);assert Path(e['image']).is_file()
assert len(data['fixed_examples'])==6
backend=LocalTransformersModel(name='cpu-check',path=P['model_path'],temperature=0.,local_files_only=True);model=CoordinateBoundary(backend);model.scope='cpu'
checks={}
with tempfile.TemporaryDirectory(dir=R/'tmp') as d:
 for arm in ['fresh','resume','B','F','H']:
  out=Path(d)/arm;out.mkdir();pipe=build(R,out,model,arm,P,old)
  assert backend._model is None
  if arm=='fresh':assert pipe.memory.summary()=={'raw':0,'abstract':0,'anchor':72,'skill':0}
  if arm in ('H','resume'):assert pipe.memory.snapshot()==old['memory']
  if arm in ('fresh','resume'):
   baseline={'current_task_metrics':{'primary_score':10/48,'correct':10,'format_compliance_rate':1.}}
   tests=[(11,2,46/48,True),(10,2,46/48,False),(11,3,46/48,False),(11,2,45/48,False)]
   for correct,loss,fmt,expected in tests:
    candidate={'current_task_metrics':{'primary_score':correct/48,'correct':correct,'format_compliance_rate':fmt},'historical_anchor_metrics':{'forget_count':loss,'total':72},'metrics':{'behavior_comparable_count':120,'behavior_change_count':3}}
    passed,_=pipe._memory_candidate_decision(baseline,candidate)
    assert passed==expected,(arm,correct,loss,fmt,passed)
    picked=pipe.optimizer.select_best_candidate([{'candidate_id':'cpu_only','validation_errors':[]}],{'cpu_only':candidate},baseline)
    assert (picked is not None)==expected
   pipe.optimizer.history.clear()
   with protect_all_anchors():
    import hcl.pipeline as module
    anchors=module._historical_anchors_for_train(pipe.memory.historical_anchors(),data['train']);context=pipe._historical_anchor_context(anchors)
    assert context['ready'],context
  checks[arm]={'memory':pipe.memory.summary(),'harness_version':pipe.harness_version,'components':pipe.optimizer.components}
  for path in pipe._checkpoint_artifacts()['paths'].values():assert Path(path).is_file(),path
 # Generic learned-state restore must retain full memory/component artifacts.
 out=Path(d)/'reload';out.mkdir();p=build(R,out,model,'N',P,old,saved={'memory':old['memory'],'artifacts':old['artifacts']});assert p.memory.snapshot()==old['memory']
result={'passed':True,'checks':checks,'model_calls':0,'confirmation_labels_opened':False,'data_roles_disjoint':True,'gate_version':2,'memory_and_component_gate_boundary_checks':8};(R/'CPU_CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
