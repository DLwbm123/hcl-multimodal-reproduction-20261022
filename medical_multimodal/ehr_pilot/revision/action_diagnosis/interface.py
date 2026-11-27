"""Type-constrained simulator. Values are generated from the chart, never filled from the reference."""
import collections
import json
import os
from pathlib import Path
from urllib.parse import parse_qs,urlsplit
import torch
import a

o,v,p,d=a.o,a.v,a.p,a.d
TYPES=('FINISH','MedicationRequest','ServiceRequest')
FIELDS={'FINISH':['answer'],
        'MedicationRequest':['subject_reference','authoredOn','status','intent','coding_system','coding_code','route','dose','dose_unit','rate','rate_unit'],
        'ServiceRequest':['subject_reference','authoredOn','status','intent','priority','coding_system','coding_code','occurrenceDateTime']}


def packet(kind,b):
    obj=dict(resourceType=kind,subject=dict(reference=b['subject_reference']),authoredOn=b['authoredOn'],status=b['status'],intent=b['intent'])
    coding=dict(coding=[dict(system=b['coding_system'],code=b['coding_code'])])
    if kind=='MedicationRequest':
        amount=dict(doseQuantity=dict(value=b['dose'],unit=b['dose_unit']))
        if b.get('rate') is not None:amount['rateQuantity']=dict(value=b['rate'],unit=b.get('rate_unit'))
        obj.update(medicationCodeableConcept=coding,dosageInstruction=[dict(route=b['route'],doseAndRate=[amount])])
    else:
        obj.update(code=coding,priority=b['priority'])
        if b.get('occurrenceDateTime') is not None:obj['occurrenceDateTime']=b['occurrenceDateTime']
    return obj


@torch.inference_mode()
def choose(chat):
    query=chat+[dict(role='user',content='Choose the next action. A = FINISH, B = POST MedicationRequest, C = POST ServiceRequest. Reply with one letter.')]
    ids=p.tok.apply_chat_template(query,tokenize=True,add_generation_prompt=True,return_tensors='pt').to(p.model.device)
    options=[p.tok.encode(letter,add_special_tokens=False) for letter in 'ABC']
    assert all(len(x)==1 for x in options)
    logits=p.model(input_ids=ids,use_cache=False).logits[0,-1,[x[0] for x in options]]
    return TYPES[int(logits.argmax())]


def flat(obj,path=''):
    if isinstance(obj,dict):
        return {k:v for key,val in obj.items() for k,v in flat(val,path+'.'+str(key)).items()}
    if isinstance(obj,list):
        return {k:v for idx,val in enumerate(obj) for k,v in flat(val,path+'['+str(idx)+']').items()}
    return {path:obj}


def simulate(c,obs,card,plan,truth,ans,oracle):
    initial=p.template.format(api_base=p.BASE.rstrip('/'),functions=p.docs,context=c['context'],question=c['instruction'])
    initial+='\nPrior workflow memory (use only if relevant): '+card
    chat=[dict(role='user',content=initial),dict(role='assistant',content='GET '+obs['url']),dict(role='user',content=obs['reply']+'\n'+o.hint(plan))]
    hist=[('agent','GET '+obs['url']),('user',obs['reply'])]
    reads=[('Observation',parse_qs(urlsplit(obs['url']).query))]
    expected=list(truth[:-1])+['FINISH']
    # Construct independent validation bodies after prompting; never passed to model.
    old=d.fetch
    try:
        d.fetch=lambda u:(json.dumps({'rows':obs['rows']}),reads[0])
        ok,_,expected_posts,_=d.canonical(c,return_trace=True)
        assert ok
    finally:d.fetch=old
    actions=[];posts=[];raws=[];answer=None
    for ix in range(7):
        kind=expected[min(ix,len(expected)-1)] if oracle else choose(chat)
        actions.append(kind)
        prompt='Action: '+kind+'. Return exactly one JSON object with fields '+', '.join(FIELDS[kind])+'. Use current task values, JSON numbers where numeric, and null only for inapplicable optional fields. For FINISH, answer is the final JSON list.'
        raw=p.generate([chat+[dict(role='user',content=prompt)]],limit=383)[0]
        raws.append(raw)
        try:
            data=json.loads(raw.strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip())
            if kind=='FINISH':
                assert isinstance(data['answer'],list)
                answer=json.dumps(data['answer'])
                hist.append(('agent','FINISH('+answer+')'))
                break
            body=packet(kind,data)
            posts.append(body)
            text='POST '+p.BASE+kind+'\n'+json.dumps(body)
            reply='POST request accepted and executed successfully.'
        except (ValueError,TypeError,KeyError,AssertionError):
            text='Invalid parameter submission for '+kind
            reply='Invalid action parameters. Submit a complete object using the declared fields.'
        hist.extend([('agent',text),('user',reply)])
        chat.extend([dict(role='assistant',content=text),dict(role='user',content=reply)])
    official,hit=p.grade(c,hist,answer,reads)
    errors=[]
    for target,actual in zip(expected_posts,posts):
        target,actual=flat(target),flat(actual)
        errors.extend(k for k,val in target.items() if k not in actual or actual[k]!=val)
    return dict(hit=hit,official=official,terminal=answer is not None and json.loads(answer)==ans,
                attempted=any(x!='FINISH' for x in actions),accepted=bool(posts),
                resource_sequence=[q['resourceType'] for q in posts]==list(truth[:-1]),first=actions[0],actions=actions,
                field_errors=sorted(set(errors)),raw=raws,history=hist)


def run():
    v.setup_model()
    shard=int(os.environ['PART']);result=[]
    for c,obs,card,plan,old in list(a.cases())[shard::4]:
        truth,ans=o.reference(c,obs)
        if len(truth)==1:continue
        row=dict(id=c['id'],wrong_plan=truth!=plan)
        if row['wrong_plan']:
            row['oracle_plan']=a.capture(c,obs,card,truth,truth,ans)
        for oracle in (False,True):
            row['forced' if oracle else 'free']=simulate(c,obs,card,plan,truth,ans,oracle)
        result.append(row)
        p.save('b'+str(shard)+'.json',result)
    p.save('b'+str(shard)+'.json',result)


def report():
    rows=[r for j in range(4) for r in o.read('b'+str(j))]
    assert len(rows)==7 and sum(r['wrong_plan'] for r in rows)==3
    out={'n':len(rows),'oracle_plan_gain':sum(r['oracle_plan']['metrics']['hit'] for r in rows if r['wrong_plan'])}
    for arm in ('free','forced'):
        sub=[r[arm] for r in rows]
        out[arm]={k:sum(s[k] for s in sub) for k in ('attempted','accepted','resource_sequence','terminal','hit','official')}
        out[arm]['first']=dict(collections.Counter(s['first'] for s in sub))
        out[arm]['field_mismatch_cases']=dict(collections.Counter(key for s in sub for key in s['field_errors']))
    out['forced_vs_free']=dict(gain=sum(r['forced']['hit'] and not r['free']['hit'] for r in rows),loss=sum(r['free']['hit'] and not r['forced']['hit'] for r in rows))
    p.save('interface.json',out)


if __name__=='__main__':
    import sys
    (report if sys.argv[1]=='report' else run)()
