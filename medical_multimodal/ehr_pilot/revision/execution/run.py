"""Fit and gate on development references; paired replay of frozen workflows."""
import json
import os
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit, parse_qs
import compare as v

def read(name):
    return json.loads(Path(name+'.json').read_text())

def latest(series):
    points=[(time,value) for s in series for time,value in s['measurements']]
    return max(points)[1] if points else None

def reference(c):
    ok,rows,posts,answer=v.d.canonical(c,return_trace=True)
    ordered=sorted(rows,key=lambda r:r['effectiveDateTime'])
    vals=[r['valueQuantity']['value'] for r in ordered]
    assert ok and vals==c['source_values'] and bool(posts)==(c['group']=='write')
    return vals[-1] if vals else None, posts, answer

def fit_field(values, examples):
    first=values[0]
    if all(x=='Patient/'+e['mrn'] for x,e in zip(values,examples)):
        return {'bind':'patient'}
    if all(x==first for x in values):
        return {'literal':first}
    if isinstance(first,list) and all(isinstance(x,list) and len(x)==len(first) for x in values):
        return {'items':[fit_field([x[i] for x in values],examples) for i in range(len(first))]}
    if isinstance(first,dict) and all(isinstance(x,dict) and x.keys()==first.keys() for x in values):
        return {'fields':{k:fit_field([x[k] for x in values],examples) for k in first}}
    if all(isinstance(x,(int,float)) for x in values):
        xs=[e['value'] for e in examples]
        i=next(i for i,x in enumerate(xs) if x!=xs[0])
        slope=(values[i]-first)/(xs[i]-xs[0])
        offset=first-slope*xs[0]
        assert all(round(slope*x+offset,3)==y for x,y in zip(xs,values))
        return {'affine':[slope,offset]}
    raise ValueError('unsupported varying field')

def render(node,mrn,value):
    if 'fields' in node:
        return {k:render(n,mrn,value) for k,n in node['fields'].items()}
    if 'items' in node:
        return [render(n,mrn,value) for n in node['items']]
    if 'bind' in node:
        return 'Patient/'+mrn
    if 'affine' in node:
        slope,offset=node['affine']
        return round(slope*value+offset,3)
    return node['literal']

def wants_write(policy,x):
    if x is None:
        return policy['missing_write']
    return (x<policy['threshold']) == policy['low_write']

def fit(cases):
    examples=[]
    for c in cases:
        value,posts,answer=reference(c)
        examples.append(dict(value=value,posts=posts,answer=answer,mrn=c['eval_MRN']))
    values=sorted({e['value'] for e in examples if e['value'] is not None})
    cuts=[(a+b)/2 for a,b in zip(values,values[1:])]
    def error(pair):
        cut,low=pair
        return sum(((e['value']<cut)==low)!=bool(e['posts']) for e in examples if e['value'] is not None)
    cut,low=min(((x,low) for x in cuts for low in (True,False)),key=error)
    missing=[bool(e['posts']) for e in examples if e['value'] is None]
    policy=dict(threshold=cut,low_write=low,missing_write=sum(missing)>len(missing)/2)
    assert all(e['answer']==examples[0]['answer'] for e in examples)
    policy['answer']=examples[0]['answer']
    positives=[e for e in examples if e['posts']]
    try:
        policy['template']=fit_field([e['posts'] for e in positives],positives)
    except (StopIteration,ValueError,AssertionError):
        policy['template']=None
    return policy

def build_gate(cases):
    v.p.ref=v.p.scorer()
    policy=fit([c for c in cases if c['stage']=='build'])
    routing=complete=0
    for c in [c for c in cases if c['stage']=='gate']:
        x,posts,answer=reference(c)
        action=wants_write(policy,x)
        routing+=action==bool(posts)
        pred=render(policy['template'],c['eval_MRN'],x) if action and policy['template'] is not None else []
        complete+=pred==posts and answer==policy['answer']
    policy.update(route_enabled=routing==24,write_enabled=routing==complete==24 and policy['template'] is not None)
    v.p.save('policy.json',policy)
    v.p.save('gate.json',dict(build=48,gate=24,routing=routing,complete=complete,route_enabled=policy['route_enabled'],write_enabled=policy['write_enabled']))

def replay(c,row,policy):
    h=row['history']
    fallback=(row['qualified'],any(r=='agent' and s.startswith('POST ') for r,s in h))
    results={'old':fallback}
    rest=('stop','route','full')
    if len(h)<2 or not h[0][1].startswith('GET ') or not h[1][1].startswith('{"returned":'):
        return dict.fromkeys(('old',*rest),fallback),False
    u=urlsplit(h[0][1][4:])
    q=parse_qs(u.query)
    if u.path!='/fhir/Observation' or q.get('patient')!=[c['eval_MRN']] or q.get('code')!=['K']:
        return dict.fromkeys(('old',*rest),fallback),False
    x=latest(json.loads(h[1][1])['series'])
    action=wants_write(policy,x)
    for arm in rest:
        if arm!='stop' and (not policy['route_enabled'] or (action and (arm=='route' or not policy['write_enabled']))):
            results[arm]=fallback
            continue
        posts=render(policy['template'],c['eval_MRN'],x) if action and arm=='full' else []
        actions=list(h[:2])
        for post in posts:
            actions += [('agent','POST '+v.p.BASE+post['resourceType']+'\n'+json.dumps(post)),('user','POST request accepted and executed successfully.')]
        answer=json.dumps(policy['answer'])
        actions.append(('agent','FINISH('+answer+')'))
        assert sum(r=='agent' for r,_ in actions)<=8
        results[arm]=(v.p.grade(c,actions,answer,[('Observation',q)])[1],bool(posts))
    return results,True

def main():
    cases=read('cases')
    test=[c for c in cases if c['stage']=='test']
    if sys.argv[1]=='build':
        build_gate(cases)
        for c in test:
            reference(c)
        v.p.save('controls.json',dict(test=len(test)))
    elif sys.argv[1]=='infer':
        assert read('controls')['test']==72
        v.setup_model()
        i=int(os.environ['PART'])
        card=read('edit_9')['candidate']
        v.p.run(test[i::3],f'old_{i}',lambda c:card,batch_size=2,hint_budget=1024)
    else:
        v.p.ref=v.p.scorer()
        rows=[row for i in range(3) for row in read(f'old_{i}')]
        by_id={r['id']:r for r in rows}
        assert len(rows)==len(by_id)==len(test)==72
        stats={}
        for c in test:
            out,eligible=replay(c,by_id[c['id']],read('policy'))
            for key in ('all',c['group']):
                s=stats.setdefault(key,Counter())
                s.update(n=1,eligible=int(eligible))
                for arm,(hit,did_write) in out.items():
                    s[arm+'_correct']+=hit
                    s[arm+'_attempt_write']+=did_write
                for a,b in [('old','full'),('old','route'),('route','full'),('stop','route')]:
                    x,y=out[a][0],out[b][0]
                    name=b+'_vs_'+a+'_'
                    s[name+'gain']+=y and not x
                    s[name+'loss']+=x and not y
                    s[name+'retained']+=x and y
        v.p.save('result.json',dict(development=read('gate'),test=stats))

if __name__=='__main__':
    main()
