import json,os
from PIL import Image
from clinical import DATA,data,split
from decision import read,visible
from observation import GLOBAL,LOCAL,observe
from shared import models,save


def run():
    cases,refs=data();_,_,test=split();choose,gen=models()
    old=read(os.environ['VETO_RUN'],'private_test.json')
    paths=read(os.environ['PAIRED_RUN'],'private_test.json')
    rows={}
    for k in test:
        r=old[k]
        if not (r['disputed'] and r['steps']>1 and not r['revised']):continue
        q=cases[k]['question'];notes=[]
        with Image.open(DATA/cases[k]['images'][0]) as src:image=src.convert('RGB')
        for n in paths[k]['baseline']['notes'][:r['steps']]:
            n=dict(n)
            if n['ok'] and n['name'] in (GLOBAL,LOCAL):
                box=json.loads(n['arg'])['box'] if n['name']==LOCAL else None
                n['text']=observe(gen,q,image,box)
            notes.append(n)
        accepted=[n for n,e in zip(notes,r['events']) if e['accepted']]
        answer=gen(visible(q,accepted)+'\nGive the concise clinical answer supported by this evidence.',96)
        correct=choose('Reference acceptance criteria: '+json.dumps(refs[k]['gt_answer'])+'\nAnswer: '+answer,{},
                       'A = answer satisfies the reference clinically including key details, B = incorrect or incomplete. Return A or B.',list('AB'))=='A'
        rows[k]=dict(notes=notes,answer=answer,correct=int(correct))
    assert len(rows)==19
    save('private_fixed',rows);save('fixed_result',dict(n=len(rows),correct=sum(x['correct'] for x in rows.values())))


if __name__=='__main__':run()
