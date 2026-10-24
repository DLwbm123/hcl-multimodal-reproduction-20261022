"""New training/report groups; membership selected before labels are associated."""
import collections
import json
import os
from pathlib import Path
import random
import time
from PIL import Image

ROOT=Path(os.environ['R981_ROOT']);BASE=Path(os.environ['R996_BASE']);PREVIOUS=Path(os.environ['R998_PREVIOUS'])


def main():
    plan=json.loads((ROOT/'code/protocol.json').read_text());frozen=ROOT/'private/members.json'
    if frozen.exists():raise RuntimeError('Members already frozen; do not resample')
    excluded=set()
    for name,rel in [('medrax_v1','native_results.private.json'),('medrax_tool_consult_v2','results.private.json'),('medrax_tool_value_v3','results.private.json'),('react_vision_v1','results.private.json')]:excluded.update(str(r['case_id']) for r in json.loads((BASE.parent/name/rel).read_text()))
    excluded.update(str(r['case_id']) for r in json.loads((BASE/'private/panel.json').read_text()))
    excluded.update(str(r['case_id']) for r in json.loads((PREVIOUS/'private/members.json').read_text()));excluded.update(str(r['case_id']) for r in json.loads((Path(os.environ['R1000_PRIOR'])/'private/members.json').read_text()));assert len(excluded)==172
    path=BASE/'data/chestagentbench/metadata.jsonl';rows=[{k:r[k] for k in ['full_question_id','case_id','images','image_source_urls','categories']} for r in map(json.loads,path.read_text().splitlines())];image_root=BASE/'data/chestagentbench/images'
    def identities(r):return {str((image_root/x).resolve()) for x in r['images']}|set(r['image_source_urls'])
    forbidden=set().union(*(identities(r) for r in rows if str(r['case_id']) in excluded))
    eligible=[r for r in rows if str(r['case_id']) not in excluded and not identities(r)&forbidden]
    b_cases={str(r['case_id']) for r in eligible if 2<=len(r['images'])<=4 and 'comparison' in r['categories']};groups={s:collections.defaultdict(list) for s in ['A','B']}
    for r in eligible:
        c=str(r['case_id']);n=len(r['images']);s='B' if c in b_cases and 2<=n<=4 and 'comparison' in r['categories'] else 'A' if c not in b_cases and n==1 and 'diagnosis' in r['categories'] else None
        if s:groups[s][c].append(r)
    rng=random.Random(plan['selection']['seed']);used=set(forbidden);members=[]
    for stage in ['A','B']:
        keys=sorted(groups[stage]);rng.shuffle(keys);selected=[]
        for case in keys:
            options=sorted(groups[stage][case],key=lambda x:x['full_question_id']);rng.shuffle(options)
            for row in options:
                if not identities(row)&used:selected.append({**row,'stage':stage,'role':'report'});used.update(identities(row));break
            if len(selected)==40:break
        assert len(selected)==40
        for i,row in enumerate(selected):row['role']='train' if i<16 else 'report'
        members.extend(selected)
    members=[members[i+j] for i in range(40) for j in [0,40]]
    assert len({str(r['case_id']) for r in members})==80;frozen.write_text(json.dumps(members,indent=2));(ROOT/'FROZEN_MEMBERS.json').write_text(json.dumps({'time':time.time(),'excluded_cases':172,'cases':80,'selection_used_labels':False}))
    lookup={r['full_question_id']:r for r in map(json.loads,path.read_text().splitlines())};inputs=[];gold={};supervision={};images=0
    for m in members:
        row=lookup[m['full_question_id']];key=m['full_question_id'];answer=str(row['answer']).strip();assert len(answer)==1 and answer in 'ABCDEF'
        paths=[str((image_root/x).resolve()) for x in m['images']]
        for f in paths:
            with Image.open(f) as im:assert getattr(im,'n_frames',1)==1;im.load();assert min(im.size)>0;images+=1
        inputs.append({'id':key,'case_id':str(m['case_id']),'stage':m['stage'],'role':m['role'],'question':row['question'],'images':paths})
        (supervision if m['role']=='train' else gold)[key]=answer
    (ROOT/'private/inputs.json').write_text(json.dumps(inputs));(ROOT/'private/sealed_report_gold.json').write_text(json.dumps(gold))
    (ROOT/'private/supervision.json').write_text(json.dumps(supervision))
    assert len(supervision)==32 and len(gold)==48 and not supervision.keys()&gold.keys()
    receipt={'time':time.time(),'cases':80,'train_cases':32,'report_cases':48,'A':40,'B':40,'planned_episodes':176,'decoded_image_occurrences':images,'case_and_known_source_overlap':0,'excluded_previous_cases':172,'selection_used_labels':False,'train_report_disjoint':True}
    (ROOT/'DATA_RECEIPT.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))

if __name__=='__main__':main()
