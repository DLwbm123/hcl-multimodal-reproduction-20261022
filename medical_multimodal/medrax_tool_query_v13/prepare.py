"""Prepare one training-only query proposal and an outcome-enriched reused B panel."""
import json,os,time
from pathlib import Path
from query_logic import FIXED_QUERY


def option_text(question,label):
    matches=list(__import__('re').finditer(r'(?m)^\s*([A-F])[.)]\s+',question));values={m.group(1):question[m.end():matches[i+1].start() if i+1<len(matches) else len(question)].strip() for i,m in enumerate(matches)};assert label in values
    return values[label]


def main():
    root=Path(os.environ['R981_ROOT']);assert not (root/'private/inputs.json').exists(),'No reset'
    v7=root.parent/'medrax_corrective_memory_v7';v9=root.parent/'medrax_learning_qualification_v9';read=lambda base,name:json.loads((base/name).read_text())
    target_pool=[x for x in read(v7,'private/inputs.json') if x['stage']=='B' and x['role']=='report'];old={r['id']:r for r in read(v7,'private/report_results.json')['H0_static'] if r['stage']=='B'};errors=[x for x in target_pool if not old[x['id']]['correct']];correct=[x for x in target_pool if old[x['id']]['correct']];assert len(target_pool)==24 and len(errors)==6
    selected=[dict(x,images=[str(Path(p).resolve()) for p in x['images']]) for x in errors[:3]+correct[:3]];assert len(selected)==6
    sources=[x for x in read(v9,'private/inputs.json') if x['stage']=='B' and x['role']=='train'];attempts={r['id']:r for r in read(v9,'private/episodes.json') if r['stage']=='B' and r['role']=='train' and r['state']=='H0_train'};feedback=read(v9,'private/supervision.json');assert len(sources)==len(attempts)==24
    source_ids={x['id'] for x in sources};target_ids={x['id'] for x in selected};assert not source_ids&target_ids
    source_images={str(Path(p).resolve()) for x in sources for p in x['images']};target_images={p for x in selected for p in x['images']};assert not source_images&target_images and all(Path(p).is_file() for p in source_images|target_images)
    reserved={x['full_question_id'] for x in read(v9,'private/members.json') if x['role']=='confirmation_reserved'};assert len(reserved)==96 and not reserved&(source_ids|target_ids)
    training=[{'source_index':i,'question':x['question'],'prior_attempt_text':attempts[x['id']]['initial_final'],'attempted_option_text':option_text(x['question'],attempts[x['id']]['predicted']),'unadjudicated_benchmark_target_text':option_text(x['question'],feedback[x['id']]),'number_of_images':len(x['images'])} for i,x in enumerate(sources)]
    for name,value in [('inputs.json',selected),('source_training.json',training),('source_ids.json',list(source_ids)),('selection.json',{'selection_uses_exposed_v7_outcomes':True,'v11_or_v12_outcomes_used_for_target_selection':False,'selected_prior_errors':3,'selected_prior_correct':3,'selected_ids_in_order':[x['id'] for x in selected]})]:(root/'private'/name).write_text(json.dumps(value,indent=2))
    (root/'FROZEN_MEMBERS.json').write_text(json.dumps({'time':time.time(),'source_cases':24,'development_cases':6,'selection':'first3 original v7B H0 errors then first3 correct in original input order','source_and_target_case_and_known_image_paths_disjoint':True,'confirmation_excluded':96}))
    gold=read(v7,'private/sealed_report_gold.json');(root/'private/sealed_development_gold.json').write_text(json.dumps({x['id']:gold[x['id']] for x in selected}))
    (root/'private/fixed_query.json').write_text(json.dumps({'query_template':FIXED_QUERY}))
    receipt={'time':time.time(),'training_B_cases':24,'training_prior_benchmark_correct':sum(attempts[x['id']]['predicted']==feedback[x['id']] for x in sources),'reused_development_B_cases':6,'prior_v7_errors':3,'prior_v7_correct_controls':3,'current_images_per_target':[len(x['images']) for x in selected],'confirmation_excluded':96,'new_case_allocation':0,'source_generation_has_no_target_inputs_or_feedback':True,'expert_adjudication_established':False};(root/'DATA_RECEIPT.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))

if __name__=='__main__':main()
