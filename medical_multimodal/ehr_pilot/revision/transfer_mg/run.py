import json
import os
import sys
from collections import Counter
from urllib.parse import parse_qs,urlsplit
import compare as v
import prior as p
import matched as m

def build(cases):
    v.p.ref=v.p.scorer()
    train=[c for c in cases if c['stage']=='build']
    assert len(train)==360
    policy=p.fit(train)
    bank=[]
    for c in train:
        x,posts,answer=p.reference(c)
        bank.append(dict(mrn=c['eval_MRN'],latest=x,posts=posts,answer=answer))
    gate=Counter()
    for c in [c for c in cases if c['stage']=='gate']:
        x,posts,answer=p.reference(c)
        write=p.wants_write(policy,x)
        near=m.nearest(bank,x)
        retrieved=m.rebound(near,c['eval_MRN'])==posts and answer==near['answer']
        predicted=p.render(policy['template'],c['eval_MRN'],x) if write and policy['template'] is not None else []
        gate.update(n=1,write=int(bool(posts)),near_write=int(bool(posts) and retrieved),near=int(retrieved),route=int(write==bool(posts)),complete=int(predicted==posts and answer==policy['answer']))
    assert gate['n']==72 and gate['write']==24
    policy.update(route_enabled=gate['route']==72,write_enabled=gate['route']==gate['complete']==72 and policy['template'] is not None)
    gate.update(template_available=int(policy['template'] is not None))
    for name,obj in [('gate',gate),('policy',policy),('bank',bank)]:
        v.p.save(name+'.json',obj)
    test=[c for c in cases if c['stage']=='test']
    for c in test:
        p.reference(c)
    assert len(test)==360
    v.p.save('controls.json',dict(test=len(test)))

def report(test):
    v.p.ref=v.p.scorer()
    gate,policy,bank=[p.read(k) for k in ('gate','policy','bank')]
    rows=[r for i in range(4) for r in p.read(f'old_{i}')]
    by_id={r['id']:r for r in rows}
    assert len(rows)==len(by_id)==len(test)==360
    stats={}
    for c in test:
        row=by_id[c['id']]
        h=row['history']
        base=(row['qualified'],any(role=='agent' and s.startswith('POST ') for role,s in h))
        scores=dict.fromkeys(('base','nearest','program','hybrid','nearest_gated','program_gated'),base)
        ok=len(h)>=2 and h[0][1].startswith('GET ') and h[1][1].startswith('{"returned":')
        if ok:
            url=urlsplit(h[0][1][4:]);q=parse_qs(url.query)
            ok=url.path=='/fhir/Observation' and q.get('code')==['MG'] and q.get('patient')==[c['eval_MRN']]
        keys=['all',c['group']]
        if c['group']=='write':
            source_x=c['source_values'][-1]
            keys.append('tier_'+('high' if source_x<1 else 'middle' if source_x<1.5 else 'low'))
        agrees=False
        if ok:
            series=json.loads(h[1][1])['series']
            assert all(s['subject']=='Patient/'+c['eval_MRN'] for s in series)
            agrees=[v for s in series for _,v in s['measurements']]==c['source_values'][::-1]
            x=p.latest(series)
            ex=m.nearest(bank,x)
            posts=m.rebound(ex,c['eval_MRN'])
            scores['nearest']=(m.grade(c,h,posts,ex['answer']),bool(posts))
            if gate['near']==72:
                scores['nearest_gated']=scores['nearest']
            if policy['route_enabled']:
                write=p.wants_write(policy,x)
                if not write or policy['write_enabled']:
                    posts=p.render(policy['template'],c['eval_MRN'],x) if write else []
                    scores['program']=(m.grade(c,h,posts,policy['answer']),bool(posts))
                    scores['hybrid']=scores['program']
                elif gate['near_write']==gate['write']:
                    scores['hybrid']=scores['nearest']
                if policy['write_enabled']:
                    scores['program_gated']=scores['program']
        for group in keys:
            s=stats.setdefault(group,Counter())
            s.update(n=1,eligible=int(ok),receipt_matches_source=int(ok and agrees),receipt_differs=int(ok and not agrees),http_400=int(len(h)>1 and h[1][1]=='FHIR HTTP 400'))
            for name,(hit,wrote) in scores.items():
                s[name+'_correct']+=hit
                s[name+'_attempt_write']+=wrote
            for b,a in [('program','base'),('program','nearest'),('hybrid','nearest'),('program_gated','nearest_gated')]:
                y,z=scores[b][0],scores[a][0]
                k=b+'_vs_'+a
                s[k+'_gain']+=y and not z;s[k+'_loss']+=z and not y;s[k+'_retained']+=z and y
    v.p.save('result.json',dict(development=gate,test=stats))

if __name__=='__main__':
    cases=p.read('cases')
    test=[c for c in cases if c['stage']=='test']
    if sys.argv[1]=='build':
        build(cases)
    elif sys.argv[1]=='infer':
        assert p.read('controls')['test']==360
        v.setup_model()
        part=int(os.environ['PART'])
        hint='To retrieve the patient’s magnesium result use GET '+v.p.BASE+'Observation?patient=<current MRN>&code=MG'
        v.p.run(test[part::4],f'old_{part}',lambda c:hint,batch_size=2,hint_budget=1024)
    else:
        report(test)
