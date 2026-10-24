"""Small meaningful checks for matched writes and symlink-safe accounting."""
import copy
import json
import os
from pathlib import Path
import tempfile
from banks import matched_banks,EMPTY
from storage import tree_bytes

inputs=[];predictions=[];records=[];gold={}
for stage in ['A','B']:
    for i in range(5):
        key=stage+str(i);answer='A' if (i<3 if stage=='A' else True) else 'B'
        inputs.append({'id':key,'role':'train','stage':stage});predictions.append({'id':key,'status':'completed','predicted':answer});gold[key]='A'
        records.append({'internal_source':{'task_id':key},'response':answer,'feedback':'A'})
before=copy.deepcopy(records);states,audit=matched_banks(records,inputs,predictions,gold)
assert states['H0_static']==EMPTY and records==before
assert audit['A']['matched_selected_per_bank']==2 and audit['B']['matched_selected_per_bank']==0
assert [r['internal_source']['task_id'] for r in states['corrective_memory']['raw']]==['A3','A4']
assert [r['internal_source']['task_id'] for r in states['success_memory']['raw']]==['A0','A1']
bad=copy.deepcopy(predictions);bad[0]['predicted']=None
try:matched_banks(records,inputs,bad,gold)
except ValueError:pass
else:raise AssertionError('Invalid training output must fail closed')
with tempfile.TemporaryDirectory() as d:
    p=Path(d);root=p/'root';root.mkdir();outside=p/'outside';outside.write_bytes(b'x'*10000);(root/'data').write_bytes(b'abc');(root/'link').symlink_to(outside);alias=p/'alias';alias.symlink_to(root)
    expected=sum(x.lstat().st_size for x in [root,root/'data',root/'link'])
    assert tree_bytes(root)==tree_bytes(alias)==expected
    os.link(root/'data',root/'hardlink');assert tree_bytes(root)==sum(x.lstat().st_size for x in [root,root/'data',root/'link'])
    assert tree_bytes(root,allocated=True)==sum(x.lstat().st_blocks*512 for x in [root,root/'data',root/'link'])
root=Path(os.environ['R981_ROOT']);items=json.loads((root/'private/inputs.json').read_text());sup=json.loads((root/'private/supervision.json').read_text())
assert len(items)==80 and len({x['case_id'] for x in items})==80
assert set(sup)=={x['id'] for x in items if x['role']=='train'} and len(sup)==32
assert all(set(x)=={'id','case_id','stage','role','question','images'} for x in items)
from hcl.memory import ExperienceMemory,MemoryConfig
from worker import context
m=ExperienceMemory(MemoryConfig(record_dir=str(root/'private/cpu_memory'),raw_capacity_per_task=32,abstract_capacity_per_task=2,skill_enabled=False,anchor_selection_mode='off'));m.restore(EMPTY)
assert context(m,'Question?',set())==('',[])
for key,pred in [('one','A'),('two','B')]:m.record_training_interaction('medical_agent',{'task_id':key,'task_type':'medical_agent','split':'train','question':'Q?\nA. First\nB. Second','answer':'B'},{'task_id':key,'answer':pred})
snapshot=m.snapshot();text,ids=context(m,'Question?',{'one','two'});assert len(ids)==2 and 'First' in text and 'Second' in text and m.snapshot()==snapshot
assert any(not x['response_is_correct'] for x in snapshot['raw']) and any(x['response_is_correct'] for x in snapshot['raw'])
receipt={'passed':True,'checks':['stage_matched_write_counts','deterministic_training_only_selection','zero_error_no_fabrication','invalid_prediction_fail_closed','source_immutable','root_alias_followed_internal_links_excluded','hardlinks_counted_once','answer_free_inputs','train_only_supervision','real_hcl_correct_and_error_records','common_semantic_context','empty_H0_unchanged'],'cases':80}
(root/'CPU_CHECK.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
