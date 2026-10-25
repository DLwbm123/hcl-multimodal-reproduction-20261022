"""Freeze reused outcome-enriched development and unchanged learned experience."""
import copy,json,os,time
from pathlib import Path
from review_logic import GENERIC,review_text


def main():
    root=Path(os.environ['R981_ROOT']);prior=root.parent/'medrax_evidence_memory_v10';read=lambda n:json.loads((prior/n).read_text())
    assert not (root/'private/inputs.json').exists(),'No reset or re-selection'
    inputs=read('private/inputs.json');old=read('private/results.json');result={r['id']:r for r in old['H0_static_r1']};assert len(result)==24
    # Selection uses already-exposed v10 outcomes, never v11 outcomes or reserve cases.
    errors=[x for x in inputs if not result[x['id']]['correct']];correct=[x for x in inputs if result[x['id']]['correct']];assert len(errors)==3
    selected=errors+correct[:3];assert len(selected)==6 and len({x['id'] for x in selected})==6
    previous=read('private/states.json');states={'H0_static':copy.deepcopy(previous['H0_static']),'generic_review':copy.deepcopy(previous['H_A']),'experience_review':copy.deepcopy(previous['H_A'])}
    for i,(rec,note) in enumerate(zip(states['generic_review']['abstract'],GENERIC)):
        rec.clear();rec.update(memory_id='generic_'+str(i),memory_type='abstract',abstract_role='main_policy' if note is GENERIC[0] else 'failure_avoidance',internal_source={'task_id':'generic_control','task_name':'medical_agent','split':'train'},note=note)
    source_ids={x['item']['id'] for x in read('private/source_training.json')};assert len(source_ids)==24 and not source_ids&{x['id'] for x in selected}
    for rec in states['experience_review']['abstract']:assert set(rec['source_ids'])<=source_ids
    reserve=[r for r in json.loads((root.parent/'medrax_learning_qualification_v9/private/members.json').read_text()) if r['role']=='confirmation_reserved'];assert len(reserve)==96 and not {r['full_question_id'] for r in reserve}&(source_ids|{x['id'] for x in selected})
    for name,value in [('inputs.json',selected),('states.json',states),('selection.json',{'source_run':'v10_H0_static_r1','selection_uses_exposed_development_outcomes':True,'v11_results_used_for_selection':False,'error_ids':[x['id'] for x in errors],'correct_control_ids':[x['id'] for x in correct[:3]],'order':'three prior errors then first three prior correct in original v10 input order'})]:(root/'private'/name).write_text(json.dumps(value,indent=2))
    (root/'STATES_FROZEN.json').write_text(json.dumps({'time':time.time(),'learned_state_reused_without_regeneration':True,'source_run':'v10','development_cases':6,'original_training_source_cases':24,'current_target_feedback_used_for_memory':False}))
    gold=read('private/sealed_development_gold.json');(root/'private/sealed_development_gold.json').write_text(json.dumps({x['id']:gold[x['id']] for x in selected}))
    chars={arm:len(review_text(arm,states)) for arm in states};ratio=chars['generic_review']/chars['experience_review'];assert .8<=ratio<=1.2
    receipt={'time':time.time(),'reused_development_cases':6,'v10_prior_errors':3,'v10_prior_correct_controls':3,'new_case_allocation':0,'confirmation_excluded':96,'source_training_cases':24,'new_memory_generation_calls':0,'source_state_reused_unchanged':True,'review_context_chars':chars,'generic_to_experience_char_ratio':ratio,'selection_uses_prior_v10_outcomes':True,'selection_uses_v11_outcomes':False,'medical_adjudication_established':False}
    (root/'DATA_RECEIPT.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))

if __name__=='__main__':main()
