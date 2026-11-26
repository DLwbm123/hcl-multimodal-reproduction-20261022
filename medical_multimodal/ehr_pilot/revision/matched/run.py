"""Compare nearest full demonstration, model adaptation and frozen program."""
import json
import os
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
import compare as v
import prior as p

def records(name):
    return json.loads(Path(name+'.json').read_text())

def nearest(bank,value):
    pool=[item for item in bank if (item['latest'] is None)==(value is None)]
    return min(pool,key=lambda item:abs(item['latest']-value) if value is not None else 0)

def signature(action,reply,mrn):
    if not action.startswith('GET ') or not reply.startswith('{"returned":'):
        return False
    url=urlsplit(action[4:])
    q=parse_qs(url.query)
    return url.path=='/fhir/Observation' and q.get('patient')==[mrn] and q.get('code')==['K']

def rebound(ex,mrn):
    return json.loads(json.dumps(ex['posts']).replace('Patient/'+ex['mrn']+'"','Patient/'+mrn+'"'))

def build(cases):
    v.p.ref=v.p.scorer()
    bank=[]
    for c in [c for c in cases if c['stage']=='build']:
        x,posts,answer=p.reference(c)
        url=v.d.query(c)
        reply,read=v.d.model_fetch(url)
        assert read and x==p.latest(json.loads(reply)['series'])
        actions=['GET '+url,'Tool returned: '+reply]
        for obj in posts:
            actions += ['POST '+v.p.BASE+obj['resourceType']+'\n'+json.dumps(obj,separators=(',',':')),'Tool returned: POST request accepted and executed successfully.']
        actions.append('FINISH('+json.dumps(answer)+')')
        demo='\n'.join(actions).replace(c['eval_MRN'],'<example MRN>')
        bank.append(dict(latest=x,posts=posts,answer=answer,mrn=c['eval_MRN'],demo=demo))
    assert len(bank)==48
    policy=records('policy')
    gate=Counter()
    for c in [c for c in cases if c['stage']=='gate']:
        x,posts,answer=p.reference(c)
        ex=nearest(bank,x)
        candidate=p.render(policy['template'],c['eval_MRN'],x) if p.wants_write(policy,x) else []
        gate['nearest']+=rebound(ex,c['eval_MRN'])==posts and ex['answer']==answer
        gate['program']+=candidate==posts and policy['answer']==answer
    v.p.save('bank.json',bank)
    v.p.save('gate.json',gate)
    for c in [c for c in cases if c['stage']=='test']:
        p.reference(c)
    v.p.save('controls.json',dict(test=72))

def inference(test):
    assert records('controls')['test']==72
    v.setup_model()
    card=records('edit_9')['candidate']
    bank=records('bank')
    header='Verified complete example from a different patient. Adapt it to the current patient and actual findings.\n'
    assert all(len(v.p.tok.encode(header+e['demo'],add_special_tokens=False))<=1024 for e in bank)
    i=int(os.environ['PART'])
    plain=v.p.generate
    # The retrieval evidence is attached to its corresponding tool receipt on later turns too.
    def persistent_demo(chats,*args,**kwargs):
        prompts=[]
        for chat in chats:
            if len(chat)>=3:
                pre=chat[:3]
                action,reply=pre[1]['content'],pre[2]['content']
                url=urlsplit(action[4:]) if action.startswith('GET ') else None
                mrn=parse_qs(url.query).get('patient',[''])[0] if url else ''
                if mrn and mrn in pre[0]['content'] and signature(action,reply,mrn):
                    item=nearest(bank,p.latest(json.loads(reply)['series']))
                    chat=chat[:2]+[dict(role='user',content=reply+'\n'+header+item['demo'])]+chat[3:]
            prompts.append(chat)
        return plain(prompts,*args,**kwargs)
    for arm in ('old','model'):
        if arm=='model':
            v.p.generate=persistent_demo
        v.p.run(test[i::3],f'{arm}_{i}',lambda c:card,batch_size=2,hint_budget=1024)

def grade(c,h,posts,answer):
    actions=list(h[:2])
    for obj in posts:
        actions.extend([('agent','POST '+v.p.BASE+obj['resourceType']+'\n'+json.dumps(obj)),('user','POST request accepted and executed successfully.')])
    ans=json.dumps(answer)
    actions.append(('agent','FINISH('+ans+')'))
    return v.p.grade(c,actions,ans,[('Observation',parse_qs(urlsplit(h[0][1][4:]).query))])[1]

def report(test):
    v.p.ref=v.p.scorer()
    arms={}
    for arm in ('old','model'):
        rows=[r for i in range(3) for r in records(f'{arm}_{i}')]
        arms[arm]={r['id']:r for r in rows}
        assert len(rows)==len(arms[arm])==len(test)==72
    gate,policy,bank=records('gate'),records('policy'),records('bank')
    stats={}
    for c in test:
        a,b=[arms[k][c['id']] for k in ('old','model')]
        assert a['history'][:2]==b['history'][:2]
        h=a['history']
        ok=len(h)>=2 and signature(h[0][1],h[1][1],c['eval_MRN'])
        scores={'old':a['qualified'],'model':b['qualified'],'nearest':a['qualified'],'program':a['qualified']}
        groups=['all',c['group']]
        if ok:
            reply=json.loads(h[1][1])
            values=[val for s in reply['series'] for _,val in s['measurements']]
            assert values==c['source_values'][::-1] and all(s['subject']=='Patient/'+c['eval_MRN'] for s in reply['series'])
            x=p.latest(reply['series'])
            ex=nearest(bank,x)
            if c['group']=='write':
                groups.append('write_exact' if ex['latest']==x else 'write_novel')
            scores['nearest']=grade(c,h,rebound(ex,c['eval_MRN']),ex['answer'])
            scores['program']=p.replay(c,a,policy)[0]['full'][0]
        for k in ('nearest','program'):
            scores[k+'_gated']=scores[k] if gate[k]==24 else scores['old']
        for g in groups:
            s=stats.setdefault(g,Counter())
            s.update(n=1,eligible=int(ok))
            for arm,hit in scores.items():
                s[arm+'_correct']+=hit
            for left,right in [('nearest','program'),('model','program'),('old','program'),('nearest_gated','program_gated')]:
                x,y=scores[left],scores[right]
                prefix=right+'_vs_'+left+'_'
                s[prefix+'gain']+=y and not x
                s[prefix+'loss']+=x and not y
                s[prefix+'retained']+=x and y
    v.p.save('result.json',dict(development=gate,test=stats))

if __name__=='__main__':
    cases=records('cases')
    test=[c for c in cases if c['stage']=='test']
    if sys.argv[1]=='build':
        build(cases)
    elif sys.argv[1]=='infer':
        inference(test)
    else:
        report(test)
