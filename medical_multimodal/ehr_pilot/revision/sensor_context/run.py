import collections,datetime as dt,json,os,random,sys
from pathlib import Path
import engine as e

e.CODES={'all':{48,58,62},'morning':{58},'evening':{62}}
e.GOALS=[[('morning','recent')],[('all','recent'),('all','previous')],[('evening','recent'),('morning','recent')]]
e.TEXT=['What is the mean morning-window sensor glucose in the latest seven calendar days?',
        'How much did mean sensor glucose across all readings change from the previous seven-day period to the latest seven-day period (latest minus previous)?',
        'Over the latest seven calendar days, what is the difference between mean evening-window and morning-window sensor glucose (evening minus morning)?']
e.DOC=e.DOC.replace('diabetes event log','sensor glucose log').replace('breakfast (before breakfast), supper (before supper)','morning (06:00 to before 10:00), evening (18:00 to before 22:00)')
ARMS=('plain','flat','roles','adapt','refresh')
def msg(role,text):return dict(role=role,content=text)
def question(c):return 'Anchor date: '+dt.date.fromordinal(c['day']).isoformat()+'\nQuestion: '+e.TEXT[c['k']]

def setup():
    visits=e.read('visits');stage={};rng=random.Random(26112710)
    for typ in ('1','2'):
        ids=sorted({x['person'] for x in visits if x['type']==typ});rng.shuffle(ids)
        a=int(len(ids)*.4);b=int(len(ids)*.15)
        for j,pid in enumerate(ids):stage[pid]='build' if j<a else 'gate' if j<a+b else 'test'
    cases=[];bank={};counts={}
    for i,visit in enumerate(visits):
        data=visit['rows'];days=sorted({d for d,_,_ in data})
        for day in sorted({days[0],days[len(days)//2],days[-1]}):
            def avg(lower,upper,code=None):
                vals=[val for d,c,val in data if lower<=d<=upper and (code is None or c==code)]
                return None if not vals else sum(vals)/len(vals)
            am=avg(day-6,day,58);pm=avg(day-6,day,62);now=avg(day-6,day);prev=avg(day-13,day-7)
            for k,y in enumerate([am,None if now is None or prev is None else now-prev,None if pm is None or am is None else pm-am]):
                c=dict(id=str(len(cases)),person=visit['person'],visit=i,stage=stage[visit['person']],k=k,day=day,truth=y)
                cases.append(c)
                trace=e.canonical(c,data) # independent direct reference, validation only outside build
                if c['stage']=='build':
                    roles=[msg('user',e.DOC+'\nVerified example from a different patient.\n'+question(c))]
                    found={};key=None
                    for line in trace.splitlines():
                        role,t=line.split(': ',1)
                        roles.append(msg('assistant' if role=='Action' else 'user',t))
                        if role=='Action':key=' '.join(t.split()[1:]) if t.startswith('READ ') else None
                        elif key is not None:found[key]=json.loads(t)['count']>0
                    bank[c['id']]=dict(roles=roles,flat='Verified example from a different patient.\n'+question(c)+'\n'+trace,found=found,day=day,k=k)
    for g in ('build','gate','test'):
        cs=[c for c in cases if c['stage']==g]
        counts[g]=dict(people=len({c['person'] for c in cs}),visits=len({c['visit'] for c in cs}),questions=len(cs))
    e.save('cases',cases);e.save('bank',bank);e.save('counts',counts);print(counts,flush=True)

def infer(stage,shard):
    import compare as v
    v.setup_model();bank=e.read('bank');visits=e.read('visits')
    cases=[c for c in e.read('cases') if c['stage']==stage][shard::4]
    for arm in ARMS:
        name=f'{stage}_{arm}_{shard}'
        rows=e.read(name) if Path(name+'.json').exists() else []
        assert [r['id'] for r in rows]==[c['id'] for c in cases[:len(rows)]]
        for start in range(len(rows),len(cases),6):
            batch=cases[start:start+6]
            ss=[e.Session(c,visits[c['visit']]['rows']) for c in batch]
            ranks=[sorted((i for i,x in bank.items() if x['k']==c['k']),key=lambda i:abs(bank[i]['day']-c['day'])) for c in batch]
            chosen=[r[0] for r in ranks];switched=[False]*len(batch)
            for step in range(8):
                live=[i for i,s in enumerate(ss) if not s.done]
                if not live:break
                chats=[]
                for i in live:
                    s=ss[i];c=batch[i];ex=bank[chosen[i]]
                    if arm in ('adapt','refresh') and not switched[i] and step:
                        seen={' '.join(key):n>0 for key,n in s.seen.items()}
                        if any(key in ex['found'] and ex['found'][key]!=flag for key,flag in seen.items()):
                            alternate=next((j for j in ranks[i] if j!=chosen[i] and all(bank[j]['found'].get(key)==flag for key,flag in seen.items())),None)
                            if alternate is not None:
                                chosen[i]=alternate if arm=='adapt' else next(j for j in ranks[i] if j!=chosen[i])
                                switched[i]=True;ex=bank[chosen[i]]
                    current='Current patient. Use current tool receipts.\n'+question(c)
                    if arm=='plain':chat=[msg('user',e.DOC+'\n'+current)]
                    elif arm=='flat':chat=[msg('user',e.DOC+'\n'+ex['flat']+'\n'+current)]
                    else:chat=ex['roles']+[msg('user',current)]
                    chats.append(chat+[msg(r,t) for r,t in s.h])
                for i,text in zip(live,v.p.generate(chats,limit=64)):ss[i].step(text.strip())
            for s,change in zip(ss,switched):
                r=s.score();r['switch']=change;r['first']=len(s.h)>1 and s.h[1][1].startswith('{"handle":')
                rows.append(r)
            e.save(name,rows);print(name,len(rows),'/',len(cases),flush=True)

def load(stage,arm):
    rows=[r for i in range(4) for r in e.read(f'{stage}_{arm}_{i}')]
    return {r['id']:r for r in rows}

def gate():
    a,b=load('gate','adapt'),load('gate','roles')
    dev=[c for c in e.read('cases') if c['stage']=='gate']
    assert len(a)==len(b)==len(dev)
    decisions=[]
    for k in range(3):
        keys=[c['id'] for c in dev if c['k']==k]
        win=sum(a[i]['hit'] and not b[i]['hit'] for i in keys)
        loss=sum(b[i]['hit'] and not a[i]['hit'] for i in keys)
        decisions.append(dict(n=len(keys),win=win,loss=loss,enabled=win>0 and loss==0))
    e.save('decision',decisions)

def report():
    test=[c for c in e.read('cases') if c['stage']=='test'];decision=e.read('decision')
    arms={a:load('test',a) for a in ARMS}
    assert all(len(rows)==len(test) for rows in arms.values())
    stats={};person=collections.defaultdict(collections.Counter)
    for c in test:
        rows={a:rs[c['id']] for a,rs in arms.items()}
        rows['selected']=rows['adapt'] if decision[c['k']]['enabled'] else rows['roles']
        assert rows['adapt']['switch']==rows['refresh']['switch']
        groups=['all',('morning','weeks','windows')[c['k']],'missing' if c['truth'] is None else 'numeric']
        if rows['adapt']['switch']:groups.append('switched')
        for group in groups:
            s=stats.setdefault(group,collections.Counter());s['n']+=1
            for arm,r in rows.items():
                for key in ('hit','raw','first','calls','switch'):s[arm+'_'+key]+=r[key]
            for a,b in [('adapt','roles'),('adapt','refresh'),('selected','roles'),('roles','flat'),('roles','plain')]:
                x,y=rows[a]['hit'],rows[b]['hit'];prefix=a+'_vs_'+b
                s[prefix+'_gain']+=x and not y;s[prefix+'_loss']+=y and not x;s[prefix+'_retained']+=x and y
        for a,b in [('roles','plain'),('adapt','roles'),('selected','roles')]:
            person[c['person']][a+'_vs_'+b]+=int(rows[a]['hit'])-int(rows[b]['hit'])
    dirs={}
    for scores in person.values():
        for arm,val in scores.items():dirs.setdefault(arm,collections.Counter())['gain' if val>0 else 'loss' if val<0 else 'tie']+=1
    e.save('result',dict(counts=e.read('counts'),decision=decision,stats=stats,patient_direction=dirs))
if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='setup':setup()
    elif mode=='select':gate()
    elif mode=='report':report()
    else:infer(mode,int(os.environ['PART']))
