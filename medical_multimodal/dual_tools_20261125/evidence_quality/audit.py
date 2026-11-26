import json,os
from pathlib import Path
from PIL import Image
from clinical import DATA,data,split
from decision import read,visible
from grounding import reference_box,iou
from shared import models,save


def audit():
    cases,refs=data();_,_,test=split()
    old=read(os.environ['VETO_RUN'],'private_test.json')
    paths=read(os.environ['PAIRED_RUN'],'private_test.json')
    choose,gen=models()
    rows={}
    for number,k in enumerate((k for k in test if old[k]['disputed']),1):
        row=old[k]
        if row['steps']<2 or row['revised']:continue
        q=cases[k]['question'];gold=refs[k]['gt_answer']
        notes=paths[k]['baseline']['notes'][:row['steps']]
        events=row['events'][:row['steps']]
        assert [n['name'] for n in notes]==[e['tool'] for e in events]
        accepted=[n for n,e in zip(notes,events) if e['accepted']]
        original=gen(visible(q,accepted)+'\nGive the concise clinical answer supported by this evidence.',96)
        def grade(ans):
            return int(choose('Reference acceptance criteria: '+json.dumps(gold)+'\nAnswer: '+ans,{},
                'A = answer satisfies the reference clinically including key details, B = incorrect or incomplete. Return A or B.',list('AB'))=='A')
        def reread(items):
            return gen(visible(q,items)+'\nAnswer only the requested fact in one short phrase. Use the observed facts, resolve conflicts by specificity, and omit background.',80)
        second=reread(accepted);all_notes=reread(notes) if len(accepted)!=len(notes) else second
        args=[t['function']['arguments'] for msg in refs[k]['dialogs'] for t in msg.get('tool_calls',[])
              if t['function']['name']=='RegionAttributeDescription']
        with Image.open(DATA/cases[k]['images'][0]) as im:
            gt=[b for a in args for b,_ in [reference_box(a,im.size)] if b]
        boxes=[json.loads(n['arg'])['box'] for n in notes if n['name']=='RegionAttributeDescription' and n['ok']]
        rows[k]=dict(number=number,original=original,verified=grade(original),accepted_reread=second,
                     accepted_score=grade(second),all_reread=all_notes,all_score=grade(all_notes),
                     positive_control=grade(str(gold['whitelist'][0][0])),
                     best_iou=max((iou(b,t) for b in boxes for t in gt),default=None))
    assert len(rows)==19
    save('private_audit',rows)
    save('audit_result',dict(n=len(rows),original=sum(r['verified'] for r in rows.values()),
        reread_accepted=sum(r['accepted_score'] for r in rows.values()),reread_all=sum(r['all_score'] for r in rows.values()),
        reference_pass=sum(r['positive_control'] for r in rows.values()),geometry_n=sum(r['best_iou'] is not None for r in rows.values()),
        any_iou_half=sum(r['best_iou'] is not None and r['best_iou']>=.5 for r in rows.values())))


if __name__=='__main__':audit()
