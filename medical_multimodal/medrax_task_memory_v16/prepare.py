"""Freeze raw source A/B memories and reused outcome-enriched panels."""
import json,os,time,copy
from pathlib import Path
from task_memory import ARMS,option_text,select,context

def main():
    root=Path(os.environ['R981_ROOT']);assert not (root/'private/inputs.json').exists(),'No reset';load=lambda r,n:json.loads((r/n).read_text());v9=root.parent/'medrax_learning_qualification_v9';v10=root.parent/'medrax_evidence_memory_v10';v15=root.parent/'medrax_visual_focus_v15'
    old=load(v10,'private/results.json');old={r['id']:r for r in old['H0_static_r1']};A=[dict(x,stage='A',images=[str(Path(y).resolve()) for y in x['images']]) for x in load(v10,'private/inputs.json') if old[x['id']]['correct']][:3];B=[dict(x,stage='B') for x in load(v15,'private/inputs.json')];assert len(A)==3 and len(B)==6
    sourcesA=load(v10,'private/source_training.json');inputs9=load(v9,'private/inputs.json');attempts={r['id']:r for r in load(v9,'private/episodes.json') if r['stage']=='B' and r['role']=='train' and r['state']=='H0_train'};feedback=load(v9,'private/supervision.json');sourcesB=[{'item':x,'attempt':attempts[x['id']],'gold':feedback[x['id']]} for x in inputs9 if x['stage']=='B' and x['role']=='train'];assert len(sourcesA)==len(sourcesB)==24
    banks={}
    for stage,sources in [('A',sourcesA),('B',sourcesB)]:
        bank=[]
        for i,s in enumerate(sources):
            x=s['item'];option_text(x['question'],s['attempt']['predicted']);option_text(x['question'],s['gold']);bank.append({'memory_id':stage+'_raw_'+str(i).zfill(2),'memory_type':'raw','internal_source':{'task_id':x['id'],'task_name':'medical_agent_'+stage,'split':'train'},'question':x['question'],'response':s['attempt']['predicted'],'feedback':s['gold'],'images':[str(Path(y).resolve()) for y in x['images']]})
        banks[stage]=bank
    source_ids={x['internal_source']['task_id'] for b in banks.values() for x in b};target_ids={x['id'] for x in A+B};assert len(source_ids)==48 and len(target_ids)==9 and not source_ids&target_ids
    source_images={y for b in banks.values() for x in b for y in x['images']};target_images={y for x in A+B for y in x['images']};assert not source_images&target_images and all(Path(y).is_file() for y in source_images|target_images)
    reserved={x['full_question_id'] for x in load(v9,'private/members.json') if x['role']=='confirmation_reserved'};assert len(reserved)==96 and not reserved&(source_ids|target_ids)
    empty={'raw':[],'abstract':[],'anchor':[],'skills':[]};states={'H0_static':copy.deepcopy(empty),'H_A':{**copy.deepcopy(empty),'raw':banks['A']},'H_B':{**copy.deepcopy(empty),'raw':banks['A']+banks['B']}}
    support={}
    for x in A+B:
        support[x['id']]={a:[r['memory_id'] for r in select(states[a],x['question'],x['stage'],a)] for a in (['H_A','H_B'] if x['stage']=='A' else ARMS)}
        if x['stage']=='A':assert support[x['id']]['H_A']==support[x['id']]['H_B'] and context(select(states['H_A'],x['question'],'A','H_A'))==context(select(states['H_B'],x['question'],'A','H_B'))
    for n,v in [('inputs.json',A+B),('states.json',states),('support_selection.json',support)]: (root/'private'/n).write_text(json.dumps(v,indent=2))
    agold=load(v10,'private/sealed_development_gold.json');bgold=load(v15,'private/sealed_development_gold.json');gold={x['id']:(agold if x['stage']=='A' else bgold)[x['id']] for x in A+B};(root/'private/sealed_development_gold.json').write_text(json.dumps(gold))
    freeze={'time':time.time(),'A_sources':24,'B_sources':24,'H_A_raw_records':24,'H_B_raw_records':48,'source_question_only_top2_selection_frozen':True,'A_context_identical_by_construction':True,'retention_A_cases':3,'acquisition_B_cases':6,'confirmation_excluded':96};(root/'STATES_FROZEN.json').write_text(json.dumps(freeze,indent=2))
    receipt={**freeze,'new_cases':0,'A_selection':'first3 original v10 H0-correct cases in input order, reused exposed labels','B_selection':'same6 v15/v13 targets originally selected from exposed v7 outcomes','training_B_prior_benchmark_correct':sum(s['attempt']['predicted']==s['gold'] for s in sourcesB),'source_and_target_ID_and_known_images_disjoint':True,'source_supervision_not_clinically_adjudicated':True,'generation_calls':0,'all_source_records_retained':True};(root/'DATA_RECEIPT.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))

if __name__=='__main__':main()
