"""Reuse known development cases; no fresh A allocation or confirmation access."""
import copy,json,os,time
from pathlib import Path
from episodic import choose,option_text

def main():
    root=Path(os.environ['R981_ROOT']);prior=root.parent/'medrax_evidence_memory_v10'
    assert not (root/'private/inputs.json').exists(),'Already frozen; no reset'
    read=lambda n:json.loads((prior/n).read_text())
    inputs=read('private/inputs.json');sources=read('private/source_training.json')
    assert len(inputs)==len(sources)==24
    records=[]
    for i,s in enumerate(sources):
        x=s['item'];assert len(x['images'])==1
        option_text(x['question'],s['attempt']['predicted']);option_text(x['question'],s['gold'])
        records.append({'memory_id':'experience_'+str(i).zfill(2),'memory_type':'raw','internal_source':{'task_id':x['id'],'task_name':'medical_agent','split':'train'},'question':x['question'],'response':s['attempt']['predicted'],'feedback':s['gold'],'images':x['images']})
    assert len({r['internal_source']['task_id'] for r in records})==24
    source_ids={r['internal_source']['task_id'] for r in records};assert not source_ids&{r['id'] for r in inputs}
    source_images={str(Path(p).resolve()) for r in records for p in r['images']}
    current_images={str(Path(p).resolve()) for r in inputs for p in r['images']}
    assert not source_images&current_images and all(Path(p).is_file() for p in source_images|current_images)
    support={}
    for x in inputs:
        selected,mismatch=choose(records,x['question']);assert selected['memory_id']!=mismatch['memory_id']
        support[x['id']]={'selected':selected['memory_id'],'mismatch':mismatch['memory_id']}
    reserved=[r for r in json.loads((root.parent/'medrax_learning_qualification_v9/private/members.json').read_text()) if r['role']=='confirmation_reserved']
    assert len(reserved)==96 and not {r['full_question_id'] for r in reserved}&(source_ids|{r['id'] for r in inputs})
    empty={'raw':[],'abstract':[],'anchor':[],'skills':[]};states={'H0_static':empty,'historical_memory':{**copy.deepcopy(empty),'raw':records}}
    for name,value in [('inputs.json',inputs),('states.json',states),('support_selection.json',support)]:
        (root/'private'/name).write_text(json.dumps(value,indent=2))
    (root/'STATES_FROZEN.json').write_text(json.dumps({'time':time.time(),'source_cases':24,'development_cases_reused':24,'selection_uses_question_only':True,'development_gold_used_for_source_selection':False,'new_confirmation_cases_used':0}))
    # Copy the existing evaluator sidecar only after selection. Its labels were exposed by v10; this is development reuse, not fresh confirmation.
    gold=read('private/sealed_development_gold.json');assert set(gold)=={r['id'] for r in inputs}
    (root/'private/sealed_development_gold.json').write_text(json.dumps(gold))
    result={'time':time.time(),'training_A_cases':24,'reused_development_A_cases':24,'fresh_development_cases':0,'protected_confirmation_cases':96,'historical_images_per_active_case':1,'same_case_or_canonical_image_overlap':0,'source_selection_uses_labels':False,'distinct_selected_sources':len({v['selected'] for v in support.values()}),'metadata_membership_and_images_reused_from_v10':True,'new_image_decode_or_hash':False}
    (root/'DATA_RECEIPT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
