"""Fit-only audit of returns filtered out by the verbatim-feature requirement."""
import json, os, re
from pathlib import Path
from decision import read
from feature_policy import incumbent
from grounding import locate
from med import main
from observation import visual_tools, LOCAL
from shared import save


def audit(cases,refs,parts,context,original,policy,available,visible,answer_from,correct,choose,gen):
    fit,_,_=parts
    call=visual_tools(original,visible,choose,gen)
    skills=read(os.environ['BOUNDED_RUN'],'private_skill.json')
    bank=[json.loads(json.loads(x)['question']) for x in (Path(os.environ['OUTCOME_FIT_CACHE'])/'memory/raw.jsonl').read_text().splitlines()]
    history={r['id']:r for r in read(os.environ['FEATURE_RUN'],'private_fit.json')}
    ids=[k for k in fit if not history[k]['quote']]
    assert len(ids)==25
    result={}
    for k in ids[int(os.environ['PART'])::int(os.environ['PARTS'])]:
        q,img,path=context(k)
        notes,trusted,old=incumbent(q,img,path,call,policy,visible,answer_from,choose,gen,bank,skills)
        prior=history[k]
        cue=prior['cue']
        assert visible(q,trusted)+'\nCurrent answer: '+old+'\nProposed local check: '+cue==prior['state']
        n=call(q+'\nInspect this specific visual distinction: '+cue,img,path,notes,LOCAL)
        proposed=answer_from(q,trusted+[n]) if n['ok'] else old
        post=correct(k,proposed)
        row=dict(pre=prior['pre'],post=post,valid=n['ok'],phrases=[],old=old,answer=proposed)
        if post>prior['pre'] and n['ok']:
            text=n['text'][:450]
            spans=[s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+',text) if len(s.strip())>=4]
            for phrase in spans[:4]:
                answers=[]
                for evidence in (phrase,text.replace(phrase,'[withheld]')):
                    draft=answer_from(q,trusted+[dict(n,text=evidence)])
                    answers.append(correct(k,draft))
                row['phrases'].append(dict(text=phrase,only=answers[0],without=answers[1]))
        result[k]=row
        save('private_filtered',result)
        print('completed',len(result),flush=True)


if __name__=='__main__':
    main(locator=locate,experiment=audit)
