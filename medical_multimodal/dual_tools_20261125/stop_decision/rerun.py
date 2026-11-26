import json
import os
from collections import Counter
from pathlib import Path
from clinical import NAMES
from decision import moves,prompt,read
from stop_rule import ablate,decide
from feedback import support
from grounding import locate
from med import SEED,main
from shared import save


def run(cases,refs,split,context,call,policy,available,visible,answer_from,correct,choose,gen):
    _,gate,test=split
    prior=Path(os.environ['BOUNDED_RUN'])
    baseline=Path(os.environ['PAIRED_RUN'])
    memory=Path(os.environ['OUTCOME_FIT_CACHE'])/'memory/raw.jsonl'
    bank=[json.loads(json.loads(x)['question']) for x in memory.read_text().splitlines()]
    assert len(bank)==91 and read(prior,'SUPERVISOR_EXIT.json')['success']
    skills=read(prior,'private_skill.json')
    result={}
    for phase,ids in [('gate',gate),('test',test)]:
        old=read(prior,'private_'+phase+'.json');base=read(baseline,'private_'+phase+'.json')
        assert set(old)==set(base)==set(ids)
        rows={}
        for i,k in enumerate(ids,1):
            q,img,path=context(k)
            notes=[];trusted=[];events=[]
            answer=answer_from(q,trusted)
            proposed=policy(q,img,notes,SEED,None)
            snapshot=None
            # Both arms use the same routing and trust sequence until the new veto stops.
            # Continue the separate router-only control from that exact prefix.
            for n in range(4):
                name=NAMES[ord(proposed)-65]
                r=call(q,img,path,notes,name)
                revised=answer_from(q,trusted+[r])
                hint=support(bank,name,q,r['text'][:450])
                ctx=('Question: '+q+'\nAccepted evidence: '+visible(q,trusted).split('\nObserved:\n',1)[1]
                     +'\nNew observation from '+name+': '+r['text'][:450]
                     +'\nPrior answer: '+answer+'\nRevised answer using the observation: '+revised)
                accepted=r['ok'] and bool(r['text'].strip()) and choose(ctx,{},skills['trust']+'\n'+hint+
                         '\nA = use the observation and revised answer. B = discard both and keep prior answer. Select one.',list('AB'))=='A'
                notes.append(r)
                if accepted:
                    trusted.append(r);answer=revised
                e=dict(tool=name,accepted=accepted)
                events.append(e)
                if n==0:first_answer=answer
                proposed='E' if n==3 else policy(q,img,notes,SEED,None)
                if proposed=='E':
                    e['decision']='budget' if n==3 else 'router_stop'
                    if snapshot is None:snapshot=(answer,len(notes))
                    break
                if snapshot is not None:continue
                planned=NAMES[ord(proposed)-65]
                state,rule=prompt(q,notes,trusted,answer,accepted,planned,skills['stop'],hint)
                if n==0 and phase=='test':
                    e['replay']=ablate(choose,state,rule,answer,planned)
                stop,reason=decide(choose,gen,q,answer,trusted,planned,skills['stop'])
                e['decision']=reason
                if stop:snapshot=(answer,len(notes))
            new_answer,steps=snapshot
            matches=(first_answer==old[k]['learned']['answer'] and
                     events[0]['accepted']==old[k]['learned']['events'][0]['accepted'])
            row=dict(disputed=not old[k]['learned']['events'][0]['base_stop'],first_matches=matches,
                     baseline=base[k]['base_score']['correct'],old=old[k]['learned']['correct'],
                     revised=correct(k,new_answer),control=correct(k,answer) if answer!=new_answer else None,
                     steps=steps,base_steps=len(base[k]['baseline']['notes']),events=events)
            if row['control'] is None:row['control']=row['revised']
            rows[k]=row
            if i%5==0:print(phase,i,flush=True)
        save('private_'+phase,rows)
        def metrics(group):
            vals=lambda key:[r[key] for r in group]
            return dict(n=len(group),old=sum(vals('old')),baseline=sum(vals('baseline')),new=sum(vals('revised')),
                        router_trust=sum(vals('control')),first_matches=sum(vals('first_matches')),
                        baseline_calls=sum(vals('base_steps')),new_calls=sum(vals('steps')),
                        control_calls=sum(len(r['events']) for r in group),continued=sum(r['steps']>1 for r in group),
                        vs_old=moves(vals('old'),vals('revised')),vs_base=moves(vals('baseline'),vals('revised')),
                        vs_control=moves(vals('control'),vals('revised')),
                        stop_reasons=dict(Counter(e['decision'] for r in group for e in r['events'][:r['steps']])))
        result[phase]={g:metrics([r for r in rows.values() if g=='all' or r['disputed']==(g=='disputed')])
                      for g in ['all','disputed','other']}
        if phase=='gate':result['accepted']=result[phase]['all']['new']>result[phase]['all']['baseline']
        save('result',result)


if __name__=='__main__':
    main(locator=locate,experiment=run)
