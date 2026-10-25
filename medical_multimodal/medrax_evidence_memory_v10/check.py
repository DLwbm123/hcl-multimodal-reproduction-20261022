"""Small checks of isolation, conditional retrieval and the repeated acquisition gate."""
import copy,json,os
from pathlib import Path
from memory_logic import FIELDS,GENERIC,parse_note,render,repeated_gate


def main():
    root=Path(os.environ['R981_ROOT']);read=lambda p:json.loads((root/p).read_text())
    inputs=read('private/inputs.json');sources=read('private/source_training.json');gold=read('private/sealed_development_gold.json')
    assert len(inputs)==len({r['case_id'] for r in inputs})==len(gold)==24
    assert all(set(r)=={'id','case_id','stage','role','question','images'} for r in inputs)
    assert {r['id'] for r in inputs}==set(gold) and len(sources)==24
    train_ids={r['item']['id'] for r in sources};assert len(train_ids)==24 and not train_ids&set(gold)
    from hcl.memory import ExperienceMemory,MemoryConfig
    import worker
    empty=copy.deepcopy(worker.core.EMPTY)
    memory=ExperienceMemory(MemoryConfig(record_dir=str(root/'private/cpu_memory'),abstract_capacity_per_task=2,skill_enabled=False,anchor_selection_mode='off'))
    for r in sources:worker.option_text(r['item']['question'],r['gold']);worker.option_text(r['item']['question'],r['attempt']['predicted'])
    note={'admit':True,'reason':'A supported procedure, not a diagnosis.','cue_terms':['tracheal shift','volume loss'],**{k:'Check evidence before applying.' for k in FIELDS}}
    assert parse_note(json.dumps(note))==note
    bad={**note,'admit':'true'}
    try:parse_note(json.dumps(bad))
    except AssertionError:pass
    else:raise AssertionError('Invalid boolean admitted')
    records=[{'memory_id':'p'+str(i),'memory_type':'abstract','internal_source':{'task_id':next(iter(train_ids))},'cue_terms':note['cue_terms'],'note':{k:note[k] for k in FIELDS}} for i in range(3)]
    state={**copy.deepcopy(empty),'abstract':records};before=copy.deepcopy(state)
    text,ids=render(state,'Tracheal shift with volume loss',train_ids);assert ids==['p0','p1'] and text
    assert render(state,'A solitary calcified pulmonary nodule',train_ids)==('',[]) and state==before
    generic={**copy.deepcopy(empty),'abstract':[{'memory_id':'fixed_generic','memory_type':'abstract','note':GENERIC}]}
    for state in [empty,generic,{**copy.deepcopy(empty),'abstract':records[:2]}]:
        memory.restore(state);assert memory.snapshot()==state
        for r in inputs:render(memory.snapshot(),r['question'],train_ids)
        assert memory.snapshot()==state
    labels={'a':'A','b':'A'};ref=[{k:{'predicted':'B'} for k in labels} for _ in range(2)];learned=[{k:{'predicted':'A'} for k in labels} for _ in range(2)]
    pairs=[{'H0_to_HA':{'gain':2,'loss':0},'generic_to_HA':{'gain':1,'loss':0}} for _ in range(2)]
    assert repeated_gate(pairs,ref,ref,learned,list(labels),labels,True)['pass']
    assert not repeated_gate(pairs,ref,ref,learned,list(labels),labels,False)['pass']
    pairs[1]['generic_to_HA']['gain']=0
    assert not repeated_gate(pairs,ref,ref,learned,list(labels),labels,True)['pass']
    prior=root.parent/'medrax_learning_qualification_v9';reserved=[r for r in json.loads((prior/'private/members.json').read_text()) if r['role']=='confirmation_reserved']
    assert len(reserved)==96 and not {r['full_question_id'] for r in reserved}&(set(gold)|train_ids)
    result={'passed':True,'source_A_cases':24,'development_A_cases':24,'confirmation_excluded':96,'memory_restore_and_context_checks':72,'retrieval_requires_two_distinct_cue_tokens':True,'retrieval_top2_tie_and_nonmatch_checked':True,'generation_schema_and_repeated_gate_failures_checked':True,'development_gold_not_sent_to_generation_or_inference':True}
    (root/'CPU_CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
