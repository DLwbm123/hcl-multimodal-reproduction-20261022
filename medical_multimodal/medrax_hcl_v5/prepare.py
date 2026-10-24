"""Freeze new case groups from metadata before attaching supervision."""
import collections
import json
import os
from pathlib import Path
import random
import time
from PIL import Image

ROOT=Path(os.environ['R981_ROOT']);BASE=Path(os.environ['R996_BASE'])

def main():
    plan=json.loads((ROOT/'code/protocol.json').read_text())
    member_path=ROOT/'private/members.json'
    if member_path.exists():raise RuntimeError('Preparation already frozen; do not resample')
    excluded=set()
    for name,rel in [('medrax_v1','native_results.private.json'),('medrax_tool_consult_v2','results.private.json'),('medrax_tool_value_v3','results.private.json'),('react_vision_v1','results.private.json')]:
        excluded.update(str(r['case_id']) for r in json.loads((BASE.parent/name/rel).read_text()))
    excluded.update(str(r['case_id']) for r in json.loads((BASE/'private/panel.json').read_text()))
    assert len(excluded)==60
    metadata=BASE/'data/chestagentbench/metadata.jsonl'
    # First pass deliberately projects away answers/explanations.
    rows=[{k:r[k] for k in ['full_question_id','case_id','images','image_source_urls','categories']} for r in map(json.loads,metadata.read_text().splitlines())]
    image_root=BASE/'data/chestagentbench/images'
    def identities(r):return {str((image_root/x).resolve()) for x in r['images']}|set(r['image_source_urls'])
    forbidden=set().union(*(identities(r) for r in rows if str(r['case_id']) in excluded))
    eligible=[r for r in rows if str(r['case_id']) not in excluded and not identities(r)&forbidden]
    b_cases={str(r['case_id']) for r in eligible if 2<=len(r['images'])<=4 and 'comparison' in r['categories']}
    groups={'A':collections.defaultdict(list),'B':collections.defaultdict(list)}
    for r in eligible:
        c=str(r['case_id']);n=len(r['images'])
        stage='B' if c in b_cases and 2<=n<=4 and 'comparison' in r['categories'] else 'A' if c not in b_cases and n==1 and 'diagnosis' in r['categories'] else None
        if stage:groups[stage][c].append(r)
    rng=random.Random(plan['selection']['seed']);members=[];used=set(forbidden)
    for stage in ['A','B']:
        keys=sorted(groups[stage]);rng.shuffle(keys);chosen=[]
        for case in keys:
            options=sorted(groups[stage][case],key=lambda x:x['full_question_id']);rng.shuffle(options)
            for row in options:
                if not identities(row)&used:
                    chosen.append(row);used.update(identities(row));break
            if len(chosen)==sum(plan['selection']['counts'][stage].values()):break
        assert len(chosen)==sum(plan['selection']['counts'][stage].values())
        offset=0
        for role,count in plan['selection']['counts'][stage].items():
            for row in chosen[offset:offset+count]:members.append({**row,'stage':stage,'role':role})
            offset+=count
    assert len(members)==88 and len({str(x['case_id']) for x in members})==88
    member_path.write_text(json.dumps(members,indent=2))
    (ROOT/'FROZEN_MEMBERS.json').write_text(json.dumps({'time':time.time(),'count':88,'excluded_prior_cases':60,'selection_used_labels':False,'source_case_group_counts':{s:len(v) for s,v in groups.items()}},indent=2))
    # Associate labels only after the metadata-only member file is durable.
    all_rows={r['full_question_id']:r for r in map(json.loads,metadata.read_text().splitlines())}
    inputs=[];supervision={};report_gold={};frames=0
    for member in members:
        row=all_rows[member['full_question_id']];key=row['full_question_id'];gold=str(row['answer']).strip();assert gold in 'ABCDEF' and len(gold)==1
        paths=[str((image_root/x).resolve()) for x in member['images']]
        for path in paths:
            with Image.open(path) as im:
                assert getattr(im,'n_frames',1)==1
                im.load();assert min(im.size)>0;frames+=1
        inputs.append({'id':key,'case_id':str(member['case_id']),'stage':member['stage'],'role':member['role'],'question':row['question'],'images':paths})
        (report_gold if member['role']=='report' else supervision)[key]=gold
    (ROOT/'private/inputs.json').write_text(json.dumps(inputs))
    (ROOT/'private/supervision.json').write_text(json.dumps(supervision))
    (ROOT/'private/sealed_report_gold.json').write_text(json.dumps(report_gold))
    receipt={'time':time.time(),'cases':len(inputs),'decoded_image_occurrences':frames,'role_counts':dict(collections.Counter(x['stage']+':'+x['role'] for x in inputs)),'case_overlap':0,'known_image_source_overlap':0,'all_single_frame':True,'report_labels_isolated':len(report_gold)==48,'supervision_labels':len(supervision),'selection_used_labels':False}
    (ROOT/'DATA_RECEIPT.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))

if __name__=='__main__':main()
