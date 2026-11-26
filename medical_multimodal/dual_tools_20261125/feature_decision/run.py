import json
import os
from collections import Counter
from pathlib import Path

from adapt import append
from clinical import NAMES
from decision import read, moves
from feedback import outcome, support
from grounding import locate
from med import main, SEED
from observation import visual_tools, LOCAL
from previous_learning import update_memory, words
from shared import ROOT, calls, example, save
from stop_rule import decide


BASE = 'At the submitted draft, decide whether one localized observation of a visible differentiating feature is worth the remaining call. Preserve a supported incumbent answer.'
EDITS = {
    'specific_gap': 'Inspect a local region when two interpretations remain plausible and a specific visible feature in that region could distinguish them. Stop if the answer already rests on that feature.',
    'verify_identity': 'Inspect a local region to verify an asserted identity, side, modality or finding when the current evidence lacks its distinguishing morphology, spatial landmark or readable sign. Stop for a supported global finding or a question the region cannot resolve.'
}


def incumbent(q, img, path, call, policy, visible, answer_from, choose, gen, bank, skills):
    notes, trusted = [], []
    answer = answer_from(q, trusted)
    action = policy(q, img, notes, SEED, None)
    for _ in range(4):
        name = NAMES[ord(action)-65]
        note = call(q, img, path, notes, name)
        revised = answer_from(q, trusted+[note])
        hint = support(bank, name, q, note['text'][:450])
        state = ('Question: '+q+'\nAccepted evidence: '+visible(q,trusted).split('\nObserved:\n',1)[1]
                 +'\nNew observation from '+name+': '+note['text'][:450]
                 +'\nPrior answer: '+answer+'\nRevised answer using the observation: '+revised)
        if note['ok'] and note['text'].strip() and choose(state, {}, skills['trust']+'\n'+hint+
                '\nA = use the observation and revised answer. B = discard both and keep prior answer. Select one.',list('AB'))=='A':
            trusted.append(note)
            answer = revised
        notes.append(note)
        if len(notes)==4:
            break
        action = policy(q,img,notes,SEED,None)
        if action=='E' or decide(choose,gen,q,answer,trusted,NAMES[ord(action)-65],skills['stop'])[0]:
            break
    return notes,trusted,answer


def state_for(q, trusted, answer, gen, visible):
    state = visible(q,trusted)+'\nCurrent answer: '+answer
    cue = gen(state+'\nName one observable feature and where to look that distinguishes the answer from a plausible alternative. Describe a concrete image check in one sentence. If magnification cannot add relevant evidence, output NONE.',64).strip()
    return state+'\nProposed local check: '+cue, cue


def investigate(q,img,path,notes,trusted,answer,cue,call,answer_from,choose,gen):
    local = call(q+'\nInspect this specific visual distinction: '+cue,img,path,notes,LOCAL)
    text = local['text'][:450]
    quote = ''
    if local['ok']:
        quote = gen('Question: '+q+'\nRequested visual check: '+cue+'\nLocal observation: '+text+
                    '\nCopy one short exact contiguous phrase describing the distinguishing visible morphology, orientation, landmark, text or optical pattern. Exclude inferred diagnosis and organ or class name alone. If absent return NONE.',60).strip().strip('"“”')
    if len(quote)<4 or quote.casefold()=='none' or quote not in text:
        return dict(local=local,quote='',answer=answer,prefer=False)
    revised = answer_from(q,trusted+[local])
    review = ('Question: '+q+'\nCheck: '+cue+'\nNew local evidence: '+text+'\nVerbatim feature: '+quote
              +'\nOriginal answer: '+answer+'\nAnswer with local evidence: '+revised)
    prefer = choose(review,{},'A = the quoted observable detail specifically favors the revised answer for the requested fact over the original, including resolving an unsupported answer. B = it does not, or it contradicts stronger evidence, or changes only the wording. Choose A or B.',list('AB'))=='A'
    return dict(local=local,quote=quote,answer=revised,prefer=prefer)


def examples_for(state, bank, exclude=None):
    terms=words(state)
    pool=[r for r in bank if r['id']!=exclude]
    def similarity(r):
        other=words(r['state'])
        return len(terms & other)/max(1,len(terms | other))
    near=sorted(pool,key=similarity,reverse=True)[:3]
    return json.dumps([{key:r[key] for key in ('cue','quote','effect','without','only','specific')} for r in near])


def request(choose,state,skill,neighbors):
    if state.rsplit("Proposed local check: ",1)[-1].strip().rstrip(".").upper() in ("", "NONE"):
        return False
    return choose(state,{},skill+'\nSimilar fit checks (effects under the training scorer): '+neighbors+
                  '\nA = inspect this local feature. B = stop with the current answer. Select one.',list('AB'))=='A'


def fit_features(fit,context,call,policy,visible,answer_from,correct,choose,gen):
    skills=read(os.environ['BOUNDED_RUN'],'private_skill.json')
    old_bank=[json.loads(json.loads(line)['question']) for line in (Path(os.environ['OUTCOME_FIT_CACHE'])/'memory/raw.jsonl').read_text().splitlines()]
    bank=[]
    for i,k in enumerate(fit,1):
        q,img,path=context(k)
        notes,trusted,answer=incumbent(q,img,path,call,policy,visible,answer_from,choose,gen,old_bank,skills)
        if len(notes)==4:
            continue
        state,cue=state_for(q,trusted,answer,gen,visible)
        r=investigate(q,img,path,notes,trusted,answer,cue,call,answer_from,choose,gen)
        pre=correct(k,answer)
        post=correct(k,r['answer']) if r['quote'] else pre
        dropped=alone=None
        if r['quote']:
            note=r['local']
            text=note['text'][:450]
            masked=dict(note,text=text.replace(r['quote'],'[withheld]'))
            kept=dict(note,text=r['quote'])
            dropped=correct(k,answer_from(q,trusted+[masked]))
            alone=correct(k,answer_from(q,trusted+[kept]))
        specific=bool(not pre and post and dropped==0 and alone==1)
        bank.append(dict(id=k,state=state,cue=cue,quote=r['quote'],pre=pre,post=post,prefer=r['prefer'],
                         effect=outcome(pre,post),without=dropped,only=alone,specific=specific))
        if i%5==0:print('fit',i,flush=True)
    assert bank
    directory=ROOT/'memory'
    directory.mkdir()
    records=[example(r['id'],json.dumps(r),'promote','promote' if r['specific'] else 'unconfirmed') for r in bank]
    memory=update_memory({'memories':[]},'feature',records,directory,gen,capacity=len(records))[0]
    # The retrieval display caps strings at 1200 characters. Read structured records from the native store.
    stored=[json.loads(json.loads(line)['question']) for line in (directory/'raw.jsonl').read_text().splitlines()]
    assert {r['id'] for r in stored}=={r['id'] for r in bank}
    abstract=' '.join(a.get('guidance','') for a in memory['abstract'])[:500]
    return stored,abstract


def experiment(cases,refs,parts,context,original,policy,available,visible,answer_from,correct,choose,gen):
    fit,gate,test=parts
    assert len(set(fit+gate+test))==91
    call=visual_tools(original,visible,choose,gen)
    stored,abstract=fit_features(fit,context,call,policy,visible,answer_from,correct,choose,gen)
    bank=stored
    candidates={}
    policies={}
    for name,edit in EDITS.items():
        skill=append(BASE,edit)+'\nHCL fit guidance: '+abstract
        policies[name]=skill
        gained=lost=requests=0
        for r in bank:
            ask=request(choose,r['state'],skill,examples_for(r['state'],stored,r['id']))
            requests+=ask
            if ask and r['prefer']:
                gained+=r['post']>r['pre']
                lost+=r['post']<r['pre']
        candidates[name]=dict(gain=gained,loss=lost,requests=requests)
    best=max(candidates,key=lambda n:(candidates[n]['gain']-candidates[n]['loss'],-candidates[n]['loss'],-candidates[n]['requests']))
    save('private_fit',bank)
    save('private_skill',dict(name=best,skill=policies[best]))
    result=dict(fit=dict(n=len(fit),eligible=len(bank),verified_quotes=sum(bool(r['quote']) for r in bank),
                         effects=dict(Counter(r['effect'] for r in bank)),feature_supported=sum(r['specific'] for r in bank),
                         candidates=candidates,selected=best))
    save('result',result)
    for phase,ids in [('gate',gate),('test',test)]:
        old=read(os.environ['QUALITY_RUN'],'private_'+phase+'.json')
        earlier=read(os.environ['VETO_RUN'],'private_'+phase+'.json')
        assert set(ids)==set(old)==set(earlier)
        rows={}
        for i,k in enumerate(ids,1):
            anchor=old[k]
            notes=anchor['observations']
            assert len(notes)==anchor['steps']
            trusted=[n for n,e in zip(notes,anchor['events']) if e['accepted']]
            answer=anchor['answer']
            q,img,path=context(k)
            audit=dict(asked=False,adopted=False)
            if len(notes)<4:
                state,cue=state_for(q,trusted,answer,gen,visible)
                audit['cue']=cue
                if request(choose,state,policies[best],examples_for(state,stored)):
                    r=investigate(q,img,path,notes,trusted,answer,cue,call,answer_from,choose,gen)
                    audit.update(asked=True,adopted=r['prefer'],quote=r['quote'],local=r['local'])
                    if r['prefer']:
                        answer=r['answer']
            score=correct(k,answer)
            if answer==anchor['answer']:
                assert score==anchor['revised']
            rows[k]=dict(**audit,answer=answer,correct=score,calls=len(notes)+int(audit['asked']))
            assert rows[k]['calls']<=4
            if i%10==0:print(phase,i,flush=True)
        save('private_'+phase,rows)
        summaries={}
        for name,prior,key in [('previous',old,'revised'),('earlier',earlier,'revised'),('repaired',old,'baseline')]:
            m=moves([prior[k][key] for k in ids],[rows[k]['correct'] for k in ids])
            m['retained']=m['before']-m['loss']
            summaries[name]=m
        result[phase]=dict(comparisons=summaries,requested=sum(r['asked'] for r in rows.values()),
                           adopted=sum(r['adopted'] for r in rows.values()),tool_calls=sum(r['calls'] for r in rows.values()),
                           incumbent_calls=sum(old[k]['steps'] for k in ids))
        if phase=='gate':result['gate_passed']=summaries['previous']['loss']==0 and summaries['previous']['after']>=6
        result['model_calls']=dict(calls)
        save('result',result)


if __name__=='__main__':
    main(locator=locate,experiment=experiment)
