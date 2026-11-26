"""A bounded card edit from first observable development discrepancies."""
import collections
import json
import os
from pathlib import Path
import random
import re
import sys
from urllib.parse import urlsplit, parse_qs
import compare as v

p,d = v.p,v.d
load=lambda name: json.loads(Path(name+'.json').read_text())


def prepare():
    rng=random.Random(261129)
    ids=load('partition')
    bank=load('library')
    entries=[]
    out=[]
    for stage in ['train','valid']:
        for i in ids[stage]:
            c=v.ROWS[i]
            k=v.match(c,bank)['key']
            if k not in (5,9): continue
            assert k==d.kind(c)
            out.append(dict(c,stage=stage,source='real'))
            for write in (False,True):
                key='S'+str(31000000+len(out))
                assert not any(x.get('eval_MRN')==key for x in v.ROWS.values())
                replica=dict(c,id=i+'_'+str(len(out)),eval_MRN=key,source='simulation',stage=stage)
                replica['instruction']=re.sub(r'\bS\d+\b',key,c['instruction'])
                choices=({True:[0.9,1.1,1.4,1.6,1.8],False:[2.0,2.2,2.5]} if k==5
                         else {True:[2.9,3.1,3.2,3.4],False:[3.5,3.8,4.2]})
                resources=[dict(resourceType='Patient',id=key,identifier=[dict(value=key)])]
                for j,branch in enumerate([not write,write]):
                    resources.append(dict(resourceType='Observation',id='D'+key+str(j),status='final',
                        subject=dict(reference='Patient/'+key),code=dict(coding=[dict(code=d.CODES[k])]),
                        effectiveDateTime=['2023-11-13T00:00:00+00:00','2023-11-13T09:00:00+00:00'][j],
                        valueQuantity=dict(value=rng.choice(choices[branch]),unit='mg/dL' if k==5 else 'mmol/L')))
                entries.extend(dict(request=dict(method='PUT',url=r['resourceType']+'/'+r['id']),resource=r) for r in resources)
                out.append(dict(replica,expected_write=write))
    for start in range(0,len(entries),60):
        res=p.session.post(p.BASE,json=dict(resourceType='Bundle',type='transaction',entry=entries[start:start+60]),timeout=90)
        res.raise_for_status()
        assert all(x['response']['status'].startswith(('200','201')) for x in res.json()['entry'])
    p.ref=p.scorer()
    truth={}
    for c in out:
        ok,_,posts,_=d.canonical(c,return_trace=True)
        assert ok and c.get('expected_write',bool(posts))==bool(posts)
        truth[c['id']]=posts
    assert not ({c['eval_MRN'] for c in out if c['stage']=='train'} & {c['eval_MRN'] for c in out if c['stage']=='valid'})
    p.save('cases.json',out);p.save('truth.json',truth)


def field_diff(ref,actual,path=''):
    if isinstance(ref,dict) and isinstance(actual,dict):
        for k in ref:
            sub=path+'.'+k if path else k
            if k not in actual:return sub
            diff=field_diff(ref[k],actual[k],sub)
            if diff:return diff
    elif isinstance(ref,list) and isinstance(actual,list) and len(ref)==len(actual):
        for x,y in zip(ref,actual):
            diff=field_diff(x,y,path+'[]')
            if diff:return diff
    elif ref!=actual:return path or 'root'
    return None


def earliest(c,row,ref):
    if row['qualified']:return 'success'
    hist=row['history']; read=False; pos=0
    for j,(role,s) in enumerate(hist):
        if role!='agent':continue
        if s.startswith('GET '):
            u=urlsplit(s[4:].strip()); q=parse_qs(u.query)
            good=(u.path=='/fhir/Observation' and c['eval_MRN'] in q.get('patient',[]) and
                d.CODES[d.kind(c)] in q.get('code',[]) and j+1<len(hist) and hist[j+1][1].startswith('{"returned":'))
            if not read and not good:return 'initial GET does not retrieve required scoped evidence'
            read=read or good
        elif s.startswith('POST '):
            if not read:return 'write before scoped evidence'
            if pos>=len(ref):return 'unexpected write for observed branch'
            if s.splitlines()[0][5:].strip()!=p.BASE+ref[pos]['resourceType']:return 'wrong write resource or order'
            try:payload=json.loads(s.partition('\n')[2])
            except ValueError:return 'unparseable write JSON'
            diff=field_diff(ref[pos],payload)
            if diff:return 'write '+str(pos+1)+' differs at '+diff
            pos+=1
        elif s.startswith('FINISH('):
            if not read:return 'finish before scoped evidence'
            if pos<len(ref):return 'finish before required write '+str(pos+1)
        else:return 'invalid action syntax'
    if not read:return 'never retrieved scoped evidence'
    if pos<len(ref):return 'missing required write '+str(pos+1)
    return 'completion or other scorer mismatch'


def execute(cases,tag,hint):
    return p.run(cases,tag,(lambda c:hint) if hint else None,batch_size=2,hint_budget=1024)


def experiment(stage,k):
    v.setup_model()
    bank=load('library');truth=load('truth')
    old=next(e['card'] for e in bank if e['key']==k) if k in load('gate')['enabled'] else ''
    cases=[c for c in load('cases') if c['stage']==stage and d.kind(c)==k]
    s=str(k)
    rows=execute(cases,stage+'_baseline_'+s,old)
    if stage=='train':
        details=[];counter=collections.Counter()
        for c,r in zip(cases,rows):
            reason=earliest(c,r,truth[c['id']])
            key=c['source']+'/'+('write' if truth[c['id']] else 'no_write')+'/'+reason
            counter[key]+=1
            details.append(dict(id=c['id'],reason=reason))
        p.save('audit_'+s+'.json',details)
        p.save('counts_'+s+'.json',counter)
        # Only a training simulation supplies the parameterized positive action schema.
        positive=next(c for c in cases if c['source']=='simulation' and truth[c['id']])
        pattern=v.material(positive)['body']
        prompt=('Write one short operational addendum to the existing workflow, targeting the observed first failures. '
                'Return only instructions that the acting agent can apply, with exact action syntax if useful. '
                'Keep both action and no-action branches safe. Use current patient observations. No example-specific IDs, '
                'no invented evidence. At most 170 words.\nExisting memory: '+(old or '(none)')+
                '\nTask: '+v.prompt_question(positive)+'\nRules: '+positive['context']+
                '\nTraining first-discrepancy counts by origin and branch: '+json.dumps(counter)+
                '\nVerified training action structure: '+pattern)
        edit=p.generate([[dict(role='user',content=prompt)]],limit=256)[0].strip()
        assert edit and not re.search(r'\bS\d+\b',edit)
        candidate=(old+'\n'+edit).strip()
        assert len(p.tok.encode(candidate,add_special_tokens=False))<=1024
        p.save('edit_'+s+'.json',dict(edit=edit,candidate=candidate))
    else:
        execute(cases,stage+'_candidate_'+s,load('edit_'+s)['candidate'])


def report():
    truths=load('truth');cases={c['id']:c for c in load('cases')}
    result={}
    for k in (5,9):
        old,new=[load('valid_'+arm+'_'+str(k)) for arm in ('baseline','candidate')]
        expected=[c['id'] for c in cases.values() if c['stage']=='valid' and d.kind(c)==k]
        assert [r['id'] for r in old]==[r['id'] for r in new]==expected
        groups=collections.defaultdict(collections.Counter)
        for a,b in zip(old,new):
            i=a['id'];g=groups[cases[i]['source']+('/write' if truths[i] else '/no_write')]
            x,y=bool(a['qualified']),bool(b['qualified'])
            g.update(n=1,old=int(x),new=int(y),gain=int(y and not x),loss=int(x and not y),retained=int(x and y))
        adopt=groups.get('real/write',{}).get('gain',0)>0 and all(not g['loss'] for g in groups.values())
        result[k]=dict(groups=groups,adopt=adopt,training=load('counts_'+str(k)))
    p.save('result.json',result)


if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='report':report()
    else:experiment(sys.argv[1],int(os.environ['WORKFLOW']))
