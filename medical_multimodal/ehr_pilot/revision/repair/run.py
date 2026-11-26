import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit, parse_qs
import compare as v
import matched as m
import prior as p

def post_class(ex):
    return json.dumps(ex['posts'],sort_keys=True,separators=(',',':')).replace('Patient/'+ex['mrn']+'"','Patient/<current>"')

def fit(rows):
    table={}
    for row in rows:
        x,code=row['latest'],post_class(row)
        label,count=table.get(x,(code,0))
        if label!=code:
            return None
        table[x]=(label,count+1)
    groups=[]
    for x,(label,n) in sorted(table.items()):
        if groups and label==groups[-1]['label']:
            groups[-1]['hi']=x
            groups[-1]['n']+=n
        else:
            groups.append(dict(lo=x,hi=x,label=label,n=n))
    if not 1<=len(groups)<=4 or min(g['n'] for g in groups)<2:
        return None
    return dict(cuts=[(a['hi']+b['lo'])/2 for a,b in zip(groups,groups[1:])],labels=[g['label'] for g in groups])

def predict(tree,x):
    return tree['labels'][sum(x>=cut for cut in tree['cuts'])]

def build(cases):
    v.p.ref=v.p.scorer()
    bank=p.read('bank')
    train=[c for c in cases if c['stage']=='build']
    assert len(train)==len(bank)==360
    forms=set()
    for c,ex in zip(train,bank):
        assert c['eval_MRN']==ex['mrn']
        u=v.d.query(c)
        reply,read=v.d.model_fetch(u)
        assert read and p.latest(json.loads(reply)['series'])==ex['latest']
        forms.add(u.replace(c['eval_MRN'],'{patient}'))
    assert len(forms)==1
    positives=[e for e in bank if e['posts']]
    random.Random(26112706).shuffle(positives)
    cv=0
    for i in range(5):
        tree=fit([e for j,e in enumerate(positives) if j%5!=i])
        cv+=sum(tree is not None and predict(tree,e['latest'])==post_class(e) for e in positives[i::5])
    tree=fit(positives);policy=p.read('policy')
    gate=Counter()
    for c in [c for c in cases if c['stage']=='gate']:
        x,posts,ans=p.reference(c)
        write=p.wants_write(policy,x)
        guess=json.loads(predict(tree,x).replace('Patient/<current>','Patient/'+c['eval_MRN'])) if write and tree else []
        gate.update(route=int(write==bool(posts)),complete=int(guess==posts and ans==policy['answer']),n=1)
    assert gate['n']==72
    enabled=tree is not None and cv==len(positives) and gate['route']==gate['complete']==72
    v.p.save('fitted.json',dict(query=forms.pop(),tree=tree,enabled=enabled))
    v.p.save('gate.json',dict(cv=cv,cv_n=len(positives),gate=gate,enabled=enabled,segments=len(tree['labels']) if tree else 0))
    test=[c for c in cases if c['stage']=='test']
    for c in test:
        p.reference(c)
    assert len(test)==144
    v.p.save('controls.json',dict(test=len(test)))

def parsed(action):
    if not action.startswith('GET '):
        return None,{}
    url=urlsplit(action[4:].strip())
    return url,parse_qs(url.query)

def trigger(action,reply,mrn):
    url,q=parsed(action)
    return (url is not None and url.path=='/fhir/Observation' and q.get('patient')==[mrn]
            and q.get('code')==['MG'] and ('date' in q or reply.startswith('FHIR HTTP ')))

def inference(test):
    assert p.read('controls')['test']==144
    v.setup_model()
    raw=v.p.generate
    template=p.read('fitted')['query']
    def guarded(chats,*args,**kwargs):
        outputs=[None]*len(chats)
        for i,chat in enumerate(chats):
            if len(chat)==3:
                action,reply=chat[1]['content'],chat[2]['content']
                _,q=parsed(action)
                mrn=q.get('patient',[''])[0]
                if mrn and mrn in chat[0]['content'] and trigger(action,reply,mrn):
                    outputs[i]='GET '+template.format(patient=mrn)
        todo=[i for i,t in enumerate(outputs) if t is None]
        if todo:
            for i,text in zip(todo,raw([chats[i] for i in todo],*args,**kwargs)):
                outputs[i]=text
        return outputs
    idx=int(os.environ['PART'])
    hint='To retrieve the patient’s magnesium result use GET '+v.p.BASE+'Observation?patient=<current MRN>&code=MG'
    for arm in ('original','repair'):
        v.p.generate=raw if arm=='original' else guarded
        v.p.run(test[idx::4],f'{arm}_{idx}',lambda c:hint,batch_size=2,hint_budget=1024)

def replay(c,row,arm,cfg,bank,policy):
    h=row['history'];mrn=c['eval_MRN']
    changed=arm=='repair' and len(h)>=2 and trigger(h[0][1],h[1][1],mrn)
    k=2 if changed else 0
    if changed:
        assert h[k][1]=='GET '+cfg['query'].format(patient=mrn)
    base=row['qualified']
    scores=dict(model=base,nearest=base,segments=base)
    if len(h)<=k+1 or not h[k+1][1].startswith('{"returned":'):
        return scores,False,changed
    action,reply=h[k][1],h[k+1][1]
    url,q=parsed(action)
    if not url or url.path!='/fhir/Observation' or q.get('code')!=['MG'] or q.get('patient')!=[mrn]:
        return scores,False,changed
    series=json.loads(reply)['series']
    assert all(s['subject']=='Patient/'+mrn for s in series)
    match=[val for s in series for _,val in s['measurements']]==c['source_values'][::-1]
    x=p.latest(series);near=m.nearest(bank,x)
    def grade(posts,answer):
        history=h[:k+2].copy()
        for obj in posts:
            history.extend([('agent','POST '+v.p.BASE+obj['resourceType']+'\n'+json.dumps(obj)),('user','POST request accepted and executed successfully.')])
        ans=json.dumps(answer)
        history.append(('agent','FINISH('+ans+')'))
        assert sum(role=='agent' for role,_ in history)<=8
        return v.p.grade(c,history,ans,[('Observation',q)])[1]
    scores['nearest']=grade(m.rebound(near,mrn),near['answer'])
    if policy['route_enabled']:
        write=p.wants_write(policy,x)
        if not write or cfg['enabled']:
            posts=json.loads(predict(cfg['tree'],x).replace('Patient/<current>','Patient/'+mrn)) if write else []
            scores['segments']=grade(posts,policy['answer'])
    return scores,match,changed

def report(test):
    v.p.ref=v.p.scorer()
    cfg,bank,policy=[p.read(k) for k in ('fitted','bank','policy')]
    arms={}
    for arm in ('original','repair'):
        rows=[r for i in range(4) for r in p.read(f'{arm}_{i}')]
        arms[arm]={r['id']:r for r in rows}
        assert len(rows)==len(arms[arm])==144
    stats={}
    for c in test:
        a,b=[arms[k][c['id']] for k in arms]
        assert a['history'][:2]==b['history'][:2]
        metrics={};outcomes={}
        for arm,row in [('original',a),('repair',b)]:
            scores,match,changed=replay(c,row,arm,cfg,bank,policy)
            metrics.update({arm+'_complete_read':int(match),arm+'_retry':int(changed)})
            for method,hit in scores.items():
                name=arm+'_'+method
                outcomes[name]=hit
                metrics.update({name+'_score':int(hit),name+'_with_complete_read':int(hit and match)})
        for left,right in [('original_nearest','repair_nearest'),('repair_nearest','repair_segments'),('repair_model','repair_segments')]:
            x,y=outcomes[left],outcomes[right];k=right+'_vs_'+left
            metrics.update({k+'_gain':int(y and not x),k+'_loss':int(x and not y),k+'_retained':int(x and y)})
        keys=['all',c['group']]
        if c['group']=='write':
            val=c['source_values'][-1]
            keys.append('tier_'+('high' if val<1 else 'middle' if val<1.5 else 'low'))
        for key in keys:
            stats.setdefault(key,Counter()).update(n=1,**metrics)
    v.p.save('result.json',dict(development=p.read('gate'),test=stats))

if __name__=='__main__':
    cases=p.read('cases')
    test=[c for c in cases if c['stage']=='test']
    if sys.argv[1]=='build':
        build(cases)
    elif sys.argv[1]=='infer':
        inference(test)
    else:
        report(test)
