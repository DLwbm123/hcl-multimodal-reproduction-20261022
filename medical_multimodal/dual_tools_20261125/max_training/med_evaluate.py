"""Train from disjoint feedback only, then compare frozen policies on the same 45."""
import json
import os
import time
from collections import Counter
from pathlib import Path
from adapt import append
from decision import moves, read
from feature_policy import BASE, EDITS, examples_for, investigate, request, state_for
from grounding import locate
from med import main
from observation import visual_tools
from previous_learning import update_memory
from shared import ROOT,example,save


def learn(tag,rows,choose,gen):
    directory=ROOT/('memory_'+tag)
    directory.mkdir()
    samples=[example(r['id'],json.dumps(r),'promote','promote' if r['specific'] else 'unconfirmed') for r in rows]
    memory=update_memory({'memories':[]},'feature',samples,directory,gen,capacity=len(samples))[0]
    bank=[json.loads(json.loads(line)['question']) for line in (directory/'raw.jsonl').read_text().splitlines()]
    assert {r['id'] for r in rows}=={r['id'] for r in bank}
    abstract=' '.join(a.get('guidance','') for a in memory['abstract'])[:500]
    scores={}
    skills={}
    for name,edit in EDITS.items():
        skills[name]=append(BASE,edit)+'\nHCL fit guidance: '+abstract
        gain=loss=calls=0
        for r in bank:
            ask=request(choose,r['state'],skills[name],examples_for(r['state'],bank,r['id']))
            calls+=ask
            if ask and r['prefer']:
                gain+=r['post']>r['pre']
                loss+=r['post']<r['pre']
        scores[name]=dict(gain=gain,loss=loss,requests=calls)
    best=max(scores,key=lambda k:(scores[k]['gain']-scores[k]['loss'],-scores[k]['loss'],-scores[k]['requests']))
    info=dict(eligible=len(rows),quoted=sum(bool(r['quote']) for r in rows),specific=sum(r['specific'] for r in rows),
              effects=dict(Counter(r['effect'] for r in rows)),candidates=scores,selected=best)
    return (skills[best],bank),info


def experiment(cases,refs,parts,context,original,policy,available,visible,answer_from,correct,choose,gen):
    fit,gate,test=parts
    aux=sorted(set(cases)-set(fit+gate+test))
    original_fit=read(os.environ['FEATURE_RUN'],'private_fit.json')
    records={r['id']:r for r in original_fit}
    attempted=set(fit)
    for d in map(Path,os.environ['SHARD_ROOTS'].split(os.pathsep)):
        while not (d/'SUPERVISOR_EXIT.json').is_file():
            time.sleep(3)
        assert read(d,'SUPERVISOR_EXIT.json')['success']
        new=read(d,'private_feedback.json')
        assert attempted.isdisjoint(new['attempted'])
        attempted.update(new['attempted'])
        records.update({r['id']:r for r in new['records']})
    assert len(cases)==107 and attempted==set(cases)-set(test) and not set(records)&set(test)
    groups={'local46':fit+gate,'max62':fit+gate+aux}
    policies={}
    result={'fit':{}}
    for tag,keys in groups.items():
        policies[tag],info=learn(tag,[records[k] for k in keys if k in records],choose,gen)
        result['fit'][tag]=dict(info,available=len(keys),at_limit=len(keys)-info['eligible'])
        print(tag,result['fit'][tag]['selected'],flush=True)
    save('result',result)
    old=read(os.environ['QUALITY_RUN'],'private_test.json')
    small=read(os.environ['FEATURE_RUN'],'private_test.json')
    old_context=read(os.environ['CONTEXT_RUN'],'private_test.json')
    assert set(old)==set(small)==set(old_context)==set(test) and all(old[k]['answer']==small[k]['answer'] for k in test)
    call=visual_tools(original,visible,choose,gen)
    private={tag:{} for tag in policies}
    for i,k in enumerate(test,1):
        q,img,path=context(k)
        notes=old[k]['observations']
        trusted=[n for n,e in zip(notes,old[k]['events']) if e['accepted']]
        initial=old[k]['answer']
        local=None
        if len(notes)<4:
            state,cue=state_for(q,trusted,initial,gen,visible)
        for tag,(skill,bank) in policies.items():
            ask=len(notes)<4 and request(choose,state,skill,examples_for(state,bank))
            if ask and local is None:
                local=investigate(q,img,path,notes,trusted,initial,cue,call,answer_from,choose,gen)
            adopted=ask and local['prefer']
            answer=local['answer'] if adopted else initial
            changed=answer!=initial
            official=correct(k,answer) if changed else small[k]['correct']
            ctx=old_context[k]['new']
            if changed:
                prompt='Question: '+q+'\nReference acceptance criteria: '+json.dumps(refs[k]['gt_answer'])+'\nAnswer: '+answer
                ctx=int(choose(prompt,{},'A = answer satisfies the reference clinically including key details, B = incorrect or incomplete. Return A or B.',list('AB'))=='A')
            private[tag][k]=dict(asked=ask,adopted=adopted,answer=answer,correct=official,context=ctx,calls=len(notes)+int(ask))
        if i%10==0:print('test',i,flush=True)
    save('private_test',private)
    result['test']={}
    for tag,rows in private.items():
        paired=moves([small[k]['correct'] for k in test],[rows[k]['correct'] for k in test])
        result['test'][tag]=dict(paired=paired,retained=paired['before']-paired['loss'],requested=sum(r['asked'] for r in rows.values()),
                                 adopted=sum(r['adopted'] for r in rows.values()),calls=sum(r['calls'] for r in rows.values()),
                                 context=moves([old_context[k]['new'] for k in test],[rows[k]['context'] for k in test]))
    result['small30']=dict(n=len(original_fit),correct=sum(r['correct'] for r in small.values()),calls=sum(r['calls'] for r in small.values()))
    save('result',result)


if __name__=='__main__':
    main(locator=locate,experiment=experiment)
