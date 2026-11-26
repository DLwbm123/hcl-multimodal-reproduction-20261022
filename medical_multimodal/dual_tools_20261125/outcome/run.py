import json
import os
from collections import Counter
from math import comb
from pathlib import Path

from clinical import NAMES,actions
from feedback import TRUST,STOP,build_memory,future,outcome,support
from grounding import locate
from med import CODES,MENU,SEED,main
from shared import calls,save


def experiment(cases,refs,split,context,call,policy,available,visible,answer_from,correct,choose,gen):
    fit,gate,test=split
    previous=Path(os.environ['PAIRED_RUN'])
    assert json.loads((previous/'SUPERVISOR_EXIT.json').read_text())['success']
    cache=os.environ.get('OUTCOME_FIT_CACHE')
    source=Path(cache) if cache else None
    if source is not None:
        rows=json.loads((source/'private_transitions.json').read_text())
        assert {r['case'] for r in rows}==set(fit)
        assert all(r['effect']==outcome(r['pre'],r['post']) for r in rows)
    else:
        rows=[]
        for i,k in enumerate(fit,1):
            q,img,path=context(k)
            notes=[]; transitions=[]
            answer=answer_from(q,notes)
            score=correct(k,answer)
            for name in actions(refs[k]):
                if len(notes)>=4:break
                if CODES[name] not in available(notes):continue
                response=call(q,img,path,notes,name)
                notes.append(response)
                revised=answer_from(q,notes)
                grade=correct(k,revised)
                transitions.append(dict(case=k,tool=name,question=q,returned=response['text'][:450],ok=response['ok'],
                                        step=len(notes),before=answer,after=revised,pre=score,post=grade,effect=outcome(score,grade)))
                answer,score=revised,grade
            for j,row in enumerate(transitions):
                row['future']=future(row['post'],[x['post'] for x in transitions[j+1:]])
            rows.extend(transitions)
            if i%5==0:print('fit',i,flush=True)
        save('private_transitions',rows)
    bank,skills=build_memory(rows,gen,choose,source)
    fit_stats={name:dict(Counter(r['effect'] for r in rows if r['tool']==name)) for name in NAMES}
    result=dict(fit_cases=len(fit),raw_records=len(bank),fit_by_tool=fit_stats,
                future=dict(Counter(r['future'] for r in rows)))
    save('result',result)
    for phase,ids in [('gate',gate),('test',test)]:
        baseline=json.loads((previous/('private_'+phase+'.json')).read_text())
        assert set(baseline)==set(ids)
        # The policy only receives question, image and actually requested tool returns.
        panel=[]
        for i,k in enumerate(ids,1):
            q,img,path=context(k)
            drafts={};proposals={};responses={}

            def draft(notes):
                key=visible(q,notes)
                if key not in drafts:drafts[key]=answer_from(q,notes)
                return drafts[key]

            def propose(notes):
                key=visible(q,notes)
                if key not in proposals:proposals[key]=policy(q,img,notes,SEED,None)
                return proposals[key]

            def episode(learned):
                notes=[]; trusted=[]; history=[]
                answer=draft(trusted)
                c=propose(notes)
                for n in range(4):
                    name=NAMES[ord(c)-65]
                    key=(json.dumps(notes,sort_keys=True),name)
                    if key not in responses:responses[key]=call(q,img,path,notes,name)
                    r=responses[key]; notes.append(r)
                    revised=draft(trusted+[r])
                    info=support(bank,name,q,r['text'][:450]) if learned else ''
                    context_text=('Question: '+q+'\nAccepted evidence: '+visible(q,trusted).split('\nObserved:\n',1)[1]
                                  +'\nNew observation from '+name+': '+r['text'][:450]
                                  +'\nPrior answer: '+answer+'\nRevised answer using the observation: '+revised)
                    trust_rule,stop_rule=skills if learned else (TRUST,STOP)
                    accept=r['ok'] and bool(r['text'].strip()) and choose(context_text,{},trust_rule+'\n'+info+
                        '\nA = use the observation and revised answer. B = discard both and keep prior answer. Select one.',list('AB'))=='A'
                    changed=revised!=answer
                    if accept:
                        trusted.append(r);answer=revised
                    event=dict(tool=name,accepted=accept,ok=r['ok'],changed_draft=changed)
                    history.append(event)
                    if n==3:
                        event['stop']='budget';break
                    proposed=propose(notes)
                    planned='finish' if proposed=='E' else NAMES[ord(proposed)-65]
                    stop=choose('Question: '+q+'\nAccepted evidence: '+visible(q,trusted)+'\nCurrent answer: '+answer
                                +'\nLast result was '+('accepted' if accept else 'discarded')+' from '+name+
                                '. Tools already tried: '+', '.join(v['name'] for v in notes)+'. Remaining calls: '+str(3-n)+
                                '. Router proposes: '+planned,{},stop_rule+'\n'+info+
                                '\nA = stop and submit current answer. B = continue acquiring evidence. Select one.',list('AB'))=='A'
                    event.update(stop=stop,base_stop=proposed=='E')
                    if stop:break
                    if proposed=='E':
                        opts=[v for v in available(notes) if v!='E']
                        proposed=choose(visible(q,notes),{'image':img},SEED+'\n'+MENU+
                                        '\nOne more observation requested. Available letters: '+' '.join(opts),opts)
                    c=proposed
                return dict(answer=answer,events=history)

            pair={arm:episode(arm=='learned') for arm in ('seed','learned')}
            pair['baseline']=dict(correct=baseline[k]['base_score']['correct'],steps=len(baseline[k]['baseline']['notes']))
            for arm in ('seed','learned'):
                pair[arm]['correct']=correct(k,pair[arm]['answer'])
            panel.append(pair)
            if i%5==0:print(phase,i,flush=True)
        summary={'n':len(ids)}
        for arm in ('baseline','seed','learned'):
            summary[arm]={'correct':sum(r[arm]['correct'] for r in panel)}
            if arm=='baseline':
                summary[arm]['tool_calls']=sum(r[arm]['steps'] for r in panel)
                continue
            events=[e for r in panel for e in r[arm]['events']]
            summary[arm].update(tool_calls=len(events),accepted=sum(e['accepted'] for e in events),
                                accepted_by_tool=dict(Counter(e['tool'] for e in events if e['accepted'])),
                                attempted_by_tool=dict(Counter(e['tool'] for e in events)),
                                stopped_before_base=sum(e['stop'] is True and not e['base_stop'] for e in events),
                                continued_past_base=sum(e['stop'] is False and e['base_stop'] for e in events))
        for arm in ('baseline','seed'):
            g=sum(r['learned']['correct']>r[arm]['correct'] for r in panel)
            l=sum(r['learned']['correct']<r[arm]['correct'] for r in panel)
            summary['vs_'+arm]=dict(gain=g,loss=l,exact_p=min(1,2*sum(comb(g+l,v) for v in range(min(g,l)+1))/2**(g+l)))
        if phase=='gate':result['deploy_candidate']=summary['learned']['correct']>summary['baseline']['correct']
        result[phase]=summary
        result['model_calls']=dict(calls)
        save('private_'+phase,dict(zip(ids,panel)))
        save('result',result)


main(locator=locate,experiment=experiment)
