"""Post-decision sensitivity to the scoring prompt and literal local-feature removal."""
import json
import os
import time
from pathlib import Path
from clinical import data, split
from decision import read, visible, moves
from shared import models, save


def main():
    cases,refs=data()
    _,gate,test=split()
    choose,gen=models()
    previous=Path(os.environ['QUALITY_RUN'])
    candidate=Path(os.environ['FEATURE_RUN'])
    output={}
    for phase,ids in [('gate',gate),('test',test)]:
        file='private_'+phase+'.json'
        while not (candidate/file).exists():
            if (candidate/'SUPERVISOR_EXIT.json').exists():
                raise RuntimeError('predictions missing after run exit')
            time.sleep(4)
        old=read(previous,file)
        new=read(candidate,file)
        context=read(os.environ['CONTEXT_RUN'],file)
        assert set(old)==set(new)==set(context)==set(ids)
        rows={}
        for k in ids:
            q=cases[k]['question']
            r=new[k]
            def grade(answer,with_question=False):
                prompt=('Question: '+q+'\n' if with_question else '')+'Reference acceptance criteria: '+json.dumps(refs[k]['gt_answer'])+'\nAnswer: '+answer
                return int(choose(prompt,{},'A = answer satisfies the reference clinically including key details, B = incorrect or incomplete. Return A or B.',list('AB'))=='A')
            changed=(r['answer']!=old[k]['answer'])
            ctx_old=context[k]['new']
            ctx_new=grade(r['answer'],True) if changed else ctx_old
            row=dict(old=ctx_old,new=ctx_new)
            if r['correct']>old[k]['revised']:
                assert changed and r['adopted'] and r['quote']
                prefix=[n for n,e in zip(old[k]['observations'],old[k]['events']) if e['accepted']]
                note=r['local']
                quote=r['quote']
                for key,tail in [('only',quote),('without',note['text'][:450].replace(quote,'[withheld]'))]:
                    draft=gen(visible(q,prefix+[dict(note,text=tail)])+'\nGive the concise clinical answer supported by this evidence.',96)
                    row[key]=grade(draft)
            rows[k]=row
        output[phase]=dict(with_question=moves([r['old'] for r in rows.values()],[r['new'] for r in rows.values()]),
                           gains_checked=sum('only' in r for r in rows.values()),
                           score_depends_on_feature=sum(r.get('only')==1 and r.get('without')==0 for r in rows.values()))
        save('private_'+phase,rows)
        save('sensitivity',output)


if __name__=='__main__':
    main()
