import gc
import json
import os
import random
import time
from pathlib import Path

import med
from feature_policy import incumbent, state_for
from grounding import locate
from observation import GLOBAL, LOCAL, visual_tools
from shared import ROOT

panel=Path(os.environ['PANEL_ROOT'])
items=json.loads((panel/'inputs.json').read_text())[int(os.environ['PART'])::6]


def read(path):
    return json.loads(path.read_text())


def write(name,data):
    p=ROOT/(name+'.json')
    tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(data))
    tmp.replace(p)


def inference(cases,refs,parts,context,original,policy,available,visible,answer_from,correct,choose,gen):
    bankfile=Path(os.environ['OUTCOME_FIT_CACHE'])/'memory/raw.jsonl'
    bank=[json.loads(json.loads(x)['question']) for x in bankfile.read_text().splitlines()]
    skills=read(Path(os.environ['BOUNDED_RUN'])/'private_skill.json')
    call=visual_tools(original,visible,choose,gen)
    rows={}
    tokens=ROOT/'tokens.jsonl'
    def mark():
        log=[json.loads(x) for x in tokens.read_text().splitlines()] if tokens.exists() else []
        return [len(log),sum(x[1] for x in log),sum(x[2] for x in log),time.monotonic()]
    def spent(start):
        return [b-a for a,b in zip(start,mark())]
    for rec in items:
        k=rec['id']
        q,img,path=context(k)
        start=mark()
        notes,trusted,s=incumbent(q,img,path,call,policy,visible,answer_from,choose,gen,bank,skills)
        costs={'S':spent(start)}
        assert 1<=len(notes)<=4
        trigger=not trusted and len(notes)<4
        start=mark()
        whole=call(q,img,path,[],GLOBAL)
        d=answer_from(q,[whole] if whole['ok'] else [])
        costs['D']=spent(start)
        answers=dict(S=s,K=s,G=s,V=s,D=d)
        raw=None; local=None; cue=None
        for arm in 'KGV':costs[arm]=[0]*4
        if trigger:
            raw=[n for n in notes if n['ok'] and n['text'].strip()]
            start=mark()
            if raw:answers['K']=answer_from(q,raw)
            costs['K']=spent(start)
            if whole['ok']:answers['G']=d
            costs['G']=costs['D']
            start=mark()
            _,cue=state_for(q,trusted,s,gen,visible)
            target=q if cue.strip().rstrip('.').upper() in ('','NONE') else q+'\nInspect this specific visual distinction: '+cue
            local=call(target,img,path,notes,LOCAL)
            if local['ok']:answers['V']=answer_from(q,[local])
            costs['V']=spent(start)
        rows[k]=dict(answers=answers,trigger=trigger,notes=notes,trusted=trusted,whole=whole,local=local,cue=cue,costs=costs)
        write('private_generated',rows)
        if len(rows)%8==0:print('generated',len(rows),flush=True)
    assert len(rows)==len(items)


def judge(model_key, output):
    from acquire import load_vlm
    os.environ['MODEL_ROOT']=os.environ[model_key]
    choose=load_vlm()
    labels=read(panel/'labels.json')
    rows=read(ROOT/'private_generated.json')
    rng=random.Random(1109+int(os.environ['PART']))
    scores={}
    for rec in items:
        k=rec['id']
        vals=sorted(set(rows[k]['answers'].values()))
        rng.shuffle(vals)
        scores[k]={}
        for a in vals:
            text='Clinical question: '+rec['q']+'\nDataset reference answer: '+labels[k]+'\nSubmitted answer: '+a
            key=choose(text,{},'Grade only the requested fact. Accept synonyms and compatible specifics. Require the requested side, count, direction, polarity and all requested parts. An explicit wrong guess, abstention, missing answer, mutually exclusive alternatives or a list that merely includes the reference fails. Ignore ancillary explanations except when contradicting the answer. A = correct. B = incorrect. C = ambiguous question/reference, unable to resolve even in principle. Select one.',list('ABC'))
            scores[k][a]=dict(A='1',B='0',C='U')[key]
        write(output,scores)


if __name__=='__main__':
    from PIL import Image
    import torch
    cases={r['id']:dict(images=[r['path']],question=r['q']) for r in items}
    med.data=lambda:(cases,{})
    med.split=lambda:([],[],list(cases))
    med.main(locator=locate,experiment=inference)
    for key,dest in [('REFEREE1','private_primary'),('REFEREE2','private_second')]:
        gc.collect()
        torch.cuda.empty_cache()
        judge(key,dest)
    write('complete',dict(n=len(items)))
