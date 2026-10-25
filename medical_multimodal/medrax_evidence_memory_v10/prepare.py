"""Allocate new A development cases without outcomes; preserve the v9 reserve."""
import json, os, random, time
from pathlib import Path
from PIL import Image


def main():
    root=Path(os.environ['R981_ROOT']);base=Path(os.environ['R996_BASE']);prior=base.parent/'medrax_learning_qualification_v9'
    assert not (root/'private/members.json').exists(),'Already allocated: no resampling'
    plan=json.loads((root/'code/protocol.json').read_text());image_root=base/'data/chestagentbench/images'
    fields=['full_question_id','case_id','images','image_source_urls','categories']
    rows=[{k:r[k] for k in fields} for r in map(json.loads,(base/'data/chestagentbench/metadata.jsonl').read_text().splitlines())]
    sources=[('medrax_v1','native_results.private.json'),('medrax_tool_consult_v2','results.private.json'),('medrax_tool_value_v3','results.private.json'),('react_vision_v1','results.private.json'),('medrax_subscription_v4','private/panel.json')]+[(n,'private/members.json') for n in ['medrax_hcl_v5','medrax_memory_interface_v6','medrax_corrective_memory_v7','medrax_error_abstraction_v8','medrax_learning_qualification_v9']]
    excluded=set()
    for name,rel in sources:excluded.update(str(r['case_id']) for r in json.loads((base.parent/name/rel).read_text()))
    assert len(excluded)==444
    identities=lambda r:{str((image_root/p).resolve()) for p in r['images']}|set(r['image_source_urls'])
    forbidden=set().union(*(identities(r) for r in rows if str(r['case_id']) in excluded))
    remaining=[r for r in rows if str(r['case_id']) not in excluded and not identities(r)&forbidden]
    b_cases={str(r['case_id']) for r in remaining if 2<=len(r['images'])<=4 and 'comparison' in r['categories']}
    groups={}
    for r in remaining:
        if str(r['case_id']) not in b_cases and len(r['images'])==1 and 'diagnosis' in r['categories']:groups.setdefault(str(r['case_id']),[]).append(r)
    rng=random.Random(plan['selection']['seed']);keys=sorted(groups);rng.shuffle(keys);members=[];used=set(forbidden)
    for key in keys:
        options=sorted(groups[key],key=lambda r:r['full_question_id']);rng.shuffle(options)
        for r in options:
            if identities(r)&used:continue
            members.append({**r,'stage':'A','role':'development'});used.update(identities(r));break
        if len(members)==24:break
    assert len(members)==24,'Insufficient independent source-disjoint A cases; no fallback'
    (root/'private/members.json').write_text(json.dumps(members,indent=2))
    (root/'FROZEN_MEMBERS.json').write_text(json.dumps({'time':time.time(),'development_cases':24,'excluded_cases_including_confirmation':444,'selection_used_labels':False}))
    ids={r['full_question_id'] for r in members}
    lookup={r['full_question_id']:r for r in map(json.loads,(base/'data/chestagentbench/metadata.jsonl').read_text().splitlines()) if r['full_question_id'] in ids}
    inputs=[];gold={}
    for m in members:
        r=lookup[m['full_question_id']];gold[r['full_question_id']]=str(r['answer']).strip();assert gold[r['full_question_id']] in list('ABCDEF')
        inputs.append({'id':r['full_question_id'],'case_id':str(r['case_id']),'stage':'A','role':'development','question':r['question'],'images':[str((image_root/p).resolve()) for p in r['images']]})
    old_inputs=json.loads((prior/'private/inputs.json').read_text());old_gold=json.loads((prior/'private/supervision.json').read_text());old_rows=json.loads((prior/'private/episodes.json').read_text())
    train=[r for r in old_inputs if r['stage']=='A' and r['role']=='train'];attempts={r['id']:r for r in old_rows if r['state']=='H0_train'}
    assert len(train)==24 and all(attempts[r['id']]['status']=='completed' and attempts[r['id']]['predicted'] in list('ABCDEF') for r in train)
    sources=[{'item':r,'attempt':attempts[r['id']],'gold':old_gold[r['id']]} for r in train]
    assert not {r['id'] for r in train}&ids
    reserved=[r for r in json.loads((prior/'private/members.json').read_text()) if r['role']=='confirmation_reserved']
    assert len(reserved)==96 and not {r['full_question_id'] for r in reserved}&(ids|{r['id'] for r in train})
    occurrences=0
    for item in inputs+train:
        for path in item['images']:
            with Image.open(path) as im:im.load();assert min(im.size)>0;occurrences+=1
    (root/'private/inputs.json').write_text(json.dumps(inputs));(root/'private/sealed_development_gold.json').write_text(json.dumps(gold));(root/'private/source_training.json').write_text(json.dumps(sources))
    receipt={'time':time.time(),'training_A_cases':24,'new_development_A_cases':24,'eligible_A_before_allocation':len(groups),'confirmation_reserved_preserved':96,'confirmation_answers_accessed':False,'case_and_known_source_overlap':0,'development_labels_sealed_after_membership':True,'decoded_image_occurrences':occurrences,'source_correct':sum(s['attempt']['predicted']==s['gold'] for s in sources),'source_benchmark_discordant':sum(s['attempt']['predicted']!=s['gold'] for s in sources)}
    (root/'DATA_RECEIPT.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))

if __name__=='__main__':main()
