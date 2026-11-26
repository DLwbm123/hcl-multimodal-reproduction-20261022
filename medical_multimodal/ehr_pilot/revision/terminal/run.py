from collections import Counter,defaultdict
import json,math,os,random,sys
from pathlib import Path
import sensor as orig
from sensor import e

def make_cases():
    visits=e.read('external')
    rows=[]
    for j,v in enumerate(visits):
        points=v['rows'];days=sorted({d for d,_,_ in points})
        for day in sorted({days[0],days[len(days)//2],days[-1]}):
            def mean(lo,hi,code=None):
                vals=[x for t,c,x in points if lo<=t<=hi and (code is None or c==code)]
                return sum(vals)/len(vals) if vals else None
            m=mean(day-6,day,58);n=mean(day-6,day,62);a=mean(day-6,day);b=mean(day-13,day-7)
            for k,x in enumerate((m,None if a is None or b is None else a-b,None if m is None or n is None else n-m)):
                c=dict(id='ext'+str(len(rows)),visit=j,person=v['person'],day=day,k=k,truth=x)
                e.canonical(c,points)
                rows.append(c)
    e.save('target',rows);print('external',len(visits),len(rows),flush=True)

def source_split():
    cs=[c for c in e.read('source/cases') if c['stage']=='build']
    ids=sorted({c['person'] for c in cs});random.Random(26112711).shuffle(ids)
    dev=set(ids[:2*len(ids)//3])
    return cs,dev

def artifacts(stage):
    src,dev=source_split();bank=e.read('source/bank')
    if stage=='gate':
        allow={c['id'] for c in src if c['person'] in dev}
        bank={i:b for i,b in bank.items() if i in allow}
        cases=[c for c in src if c['person'] not in dev]
        visits=e.read('source/visits')
    else:
        cases=e.read('target');visits=e.read('external')
    out={}
    for k in range(3):
        views=[]
        for ex in bank.values():
            if ex['k']!=k or ex['roles'][-1]['content']=='FINISH NA':continue
            roles=ex['roles'];seen=set();stats=set();handles={}
            for i,m in enumerate(roles):
                if m['role']!='assistant':continue
                cmd=m['content'].split()
                if cmd[0]=='READ':
                    key=tuple(cmd[1:]);handle=json.loads(roles[i+1]['content'])['handle']
                    handles[handle]=key;seen.add(key)
                if cmd[0]=='STAT' and cmd[-1]=='mean':
                    stats.add(handles[cmd[1]])
            views.append((seen,stats))
        assert views
        out[k]=(set.intersection(*(s for s,_ in views)),set.intersection(*(s for _,s in views)))
    return cases,visits,bank,out

def issue(s,library):
    required,mean_keys=library[s.c['k']]
    empty=any(s.seen.get(key)==0 for key in required)
    if s.value is None:
        if not empty:return 'NA has no supporting empty READ receipt for a required group.'
        return None
    if not isinstance(s.value,float) or not math.isfinite(s.value):return 'The proposed final answer cannot be parsed as a finite number or NA.'
    if empty:return 'Your numeric answer conflicts with an empty READ receipt for a required group.'
    absent=sorted(required-s.seen.keys())
    if absent:return 'A required group has no READ receipt: '+'; '.join(' '.join(q) for q in absent)+'.'
    absent=sorted(mean_keys-s.means)
    if absent:return 'A required group has no mean STAT receipt: '+'; '.join(' '.join(q) for q in absent)+'.'
    return None

def score(s):
    r=s.score()
    r['calls']=sum(role=='assistant' and not t.startswith('FINISH ') for role,t in s.h)
    r['steps']=sum(role=='assistant' for role,t in s.h)
    return r

def inference(stage):
    import compare as v
    v.setup_model()
    cases,visits,bank,library=artifacts(stage)
    i=int(os.environ['PART']);subset=cases[i::4]
    path=f'{stage}{i}';out=e.read(path) if Path(path+'.json').exists() else []
    assert [r['id'] for r in out]==[c['id'] for c in subset[:len(out)]]
    for start in range(len(out),len(subset),6):
        batch=subset[start:start+6]
        pref=[]
        for c in batch:
            ex=min((b for b in bank.values() if b['k']==c['k']),key=lambda b:abs(b['day']-c['day']))
            pref.append(ex['roles']+[orig.msg('user','Current patient. Use current tool receipts.\n'+orig.question(c))])
        def advance(ss,indices):
            for _ in range(8):
                live=[i for i in indices if not ss[i].done and sum(r=='assistant' for r,_ in ss[i].h)<8]
                if not live:break
                prompts=[pref[i]+[orig.msg(r,t) for r,t in ss[i].h] for i in live]
                for ix,t in zip(live,v.p.generate(prompts,limit=64)):ss[ix].step(t.strip())
        base=[e.Session(c,visits[c['visit']]['rows']) for c in batch];advance(base,range(len(base)))
        reasons=[issue(s,library) if s.done and score(s)['steps']<8 else None for s in base]
        records=[dict(id=c['id'],base=score(s),reason=reason) for c,s,reason in zip(batch,base,reasons)]
        for arm in ('generic','attributed'):
            ss={}
            for j,reason in enumerate(reasons):
                if not reason:
                    records[j][arm]=records[j]['base'];continue
                c=batch[j];s=e.Session(c,visits[c['visit']]['rows'])
                for role,t in base[j].h:
                    if role=='assistant':s.step(t)
                assert s.h==base[j].h
                s.done=False;s.value='invalid'
                note='Recheck the evidence for your proposed final answer. Continue with one documented action.'
                if arm=='attributed':note=reason+' '+note
                s.h.append(['user',note]);ss[j]=s
            advance(ss,list(ss))
            for ix,s in ss.items():records[ix][arm]=score(s)
        out.extend(records);e.save(path,out);print(stage,i,len(out),'/',len(subset),flush=True)

def paired(rows,src,dst):
    x=[r[src]['hit'] for r in rows];y=[r[dst]['hit'] for r in rows]
    return dict(gain=sum(b and not a for a,b in zip(x,y)),loss=sum(a and not b for a,b in zip(x,y)),retained=sum(a and b for a,b in zip(x,y)))

def summary(stage):
    cs,_,_,_=artifacts(stage);by={c['id']:c for c in cs}
    rows=[r for i in range(4) for r in e.read(f'{stage}{i}')]
    assert len(rows)==len({r['id'] for r in rows})==len(by)
    if stage=='gate':
        gate=[]
        for k in range(3):
            sub=[r for r in rows if by[r['id']]['k']==k];stat=paired(sub,'base','attributed')
            gate.append(dict(stat,n=len(sub),enabled=stat['gain']>0 and stat['loss']==0))
        e.save('decision',gate);return
    gate=e.read('decision');groups=defaultdict(list);people=defaultdict(Counter)
    for r in rows:
        c=by[r['id']];r['selected']=r['attributed'] if gate[c['k']]['enabled'] else r['base']
        for g in ('all',('morning','weeks','windows')[c['k']],'numeric' if c['truth'] is not None else 'missing','triggered' if r['reason'] else 'untriggered'):groups[g].append(r)
        for a in ('attributed','generic','selected'):people[c['person']][a]+=int(r[a]['hit'])-int(r['base']['hit'])
    stats={}
    for g,sub in groups.items():
        s=Counter(n=len(sub),triggered=sum(bool(r['reason']) for r in sub))
        for a in ('base','generic','attributed','selected'):
            for key in ('hit','raw','calls','steps'):s[a+'_'+key]=sum(r[a][key] for r in sub)
        for a,b in [('generic','attributed'),('base','attributed'),('base','generic'),('base','selected')]:
            s.update({b+'_vs_'+a+'_'+k:v for k,v in paired(sub,a,b).items()})
        stats[g]=s
    direction={a:dict(Counter('gain' if p[a]>0 else 'loss' if p[a]<0 else 'tie' for p in people.values())) for a in ('attributed','generic','selected')}
    e.save('result',dict(participants=len(people),stats=stats,gate=gate,patient_direction=direction))
if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='prepare':make_cases()
    elif mode in ('gate','test'):inference(mode)
    else:summary('gate' if mode=='select' else 'test')
