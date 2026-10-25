"""Exercise exact note schema, immutable HCL storage and the raw control."""
import json
import os
from pathlib import Path
from notes_logic import FIELDS,parse_note
from hcl.memory import ExperienceMemory,MemoryConfig
from worker import context
from run import EMPTY
from interface import option_texts

valid={k:'Observed cue and conditional correction, uncertain outside these conditions.' for k in FIELDS};assert parse_note(json.dumps(valid))==valid
for invalid in [{**valid,'extra':'bad'},{**valid,'uncertainty':''},{**valid,'corrective_rule':'x'*1801},{**valid,'corrective_rule':'Return only the answer'},{**valid,'corrective_rule':'Select option B'}]:
    try:parse_note(json.dumps(invalid))
    except ValueError:pass
    else:raise AssertionError('Invalid note accepted')
root=Path(os.environ['R981_ROOT']);source=json.loads((root/'private/source_state.json').read_text());items=json.loads((root/'private/inputs.json').read_text());train=json.loads((root/'private/source_training.json').read_text());ids={r['internal_source']['task_id'] for r in source['raw']}
assert len(items)==len({x['case_id'] for x in items})==48 and len(ids)==len(train)==2
assert all(set(x)=={'id','case_id','stage','role','question','images'} for x in items)
assert not {x['id'] for x in items}&ids and all(x['role']=='report' for x in items)
for record in source['raw']:
    mapping=option_texts(record['question']);assert record['response'] in mapping and record['feedback'] in mapping and record['response']!=record['feedback']
m=ExperienceMemory(MemoryConfig(record_dir=str(root/'private/cpu_memory'),raw_capacity_per_task=4,abstract_capacity_per_task=2,skill_enabled=False,anchor_selection_mode='off'));m.restore(EMPTY);assert context(m,'Q',ids)==('',[])
import importlib.util
spec=importlib.util.spec_from_file_location('prior_worker',Path(os.environ['R1002_SOURCE'])/'code/worker.py');prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
m.restore(source)
for item in items:assert context(m,item['question'],ids)==prior.context(m,item['question'],ids)
notes={'raw':[],'abstract':[{'memory_id':'n'+str(i),'memory_type':'abstract','internal_source':{'task_id':key},'note':valid} for i,key in enumerate(sorted(ids))],'anchor':[],'skills':[]};m.restore(notes)
text,selected=context(m,'Q',ids);assert len(selected)==2 and all(k in text for k in FIELDS) and m.snapshot()==notes
assert source==json.loads((root/'private/source_state.json').read_text())
receipt={'passed':True,'checks':['strict_schema_and_size','empty_extra_directive_rejection','all48_inputs_answer_free','source_is_exact_two_prior_training_errors','raw_context_exactly_matches_v7_all48','all_six_note_fields_exposed','HCL_snapshot_roundtrip','empty_H0_unchanged','source_immutable'],'cases':48};(root/'CPU_CHECK.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
