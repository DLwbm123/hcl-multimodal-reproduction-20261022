import datetime as dt
import json,os,random,sys
from collections import Counter
from pathlib import Path
import compare as v
import engine as e

ARMS=('plain','after','before','roles','remind','rescue')
cases=[c for c in e.read('cases') if c['stage']=='build']
patients=e.read('patients')
ids=sorted({c['person'] for c in cases});random.Random(26112709).shuffle(ids)
assert len(ids)==35 and len(cases)==210
folds={person:i//7 for i,person in enumerate(ids)}
msg=lambda role,content:dict(role=role,content=content)

def question(c):
    return 'Anchor date: '+dt.date.fromordinal(c['day']).isoformat()+'\nQuestion: '+e.TEXT[c['k']]

def example(c,f):
    src=min((s for s in cases if folds[s['person']] not in (f,(f+1)%5) and s['k']==c['k']),key=lambda s:abs(s['day']-c['day']))
    lines=e.canonical(src,patients[src['person']]).splitlines()
    note='Verified example from a different patient.\n'+question(src)
    roles=[msg('user',e.DOC+'\n'+note)]
    for line in lines:
        pre,txt=line.split(': ',1)
        roles.append(msg('assistant' if pre=='Action' else 'user',txt))
    return note+'\n'+'\n'.join(lines),roles

def input_chat(c,ex,arm,s,rescued):
    flat,roles=ex
    live='Current patient. Use current tool receipts.\n'+question(c)
    history=[msg(r,t) for r,t in s.h]
    if arm=='roles' or (arm=='rescue' and rescued):
        return roles+[msg('user',live)]+history
    content=e.DOC+'\n'+live
    if arm=='before':content=e.DOC+'\n'+flat+'\n'+live
    if arm=='after':content+='\n'+flat
    return [msg('user',content)]+history

def evaluate(f,shard):
    subset=[c for c in cases if folds[c['person']] in (f,(f+1)%5)][shard::4]
    examples={c['id']:example(c,f) for c in subset}
    for arm in ARMS:
        name=f'part_{f}_{shard}_{arm}'
        out=e.read(name) if Path(name+'.json').exists() else []
        assert [r['id'] for r in out]==[c['id'] for c in subset[:len(out)]]
        for j in range(len(out),len(subset),6):
            batch=subset[j:j+6];ss=[e.Session(c,patients[c['person']]) for c in batch]
            rescued=[False]*len(batch);first=[False]*len(batch)
            for step in range(8):
                live=[i for i,s in enumerate(ss) if not s.done]
                if not live:break
                chats=[input_chat(batch[i],examples[batch[i]['id']],arm,ss[i],rescued[i]) for i in live]
                for i,raw in zip(live,v.p.generate(chats,limit=64)):
                    s=ss[i];s.step(raw.strip())
                    if step==0:first[i]=bool(s.seen)
                    bad=s.done and not s.seen
                    invalid=s.h[-1][0]=='user' and s.h[-1][1].startswith('Invalid action.')
                    if arm in ('remind','rescue') and not rescued[i] and (invalid or bad) and step<7:
                        rescued[i]=True
                        hint='Return one executable action only. Read current evidence before finishing.'
                        if bad:
                            s.done=False;s.value='invalid';s.h.append(['user',hint])
                        else:s.h[-1][1]+=' '+hint
            for c,s,r,first_ok in zip(batch,ss,rescued,first):
                item=s.score()
                # Count submitted non-FINISH actions, not the controller reminder as a tool call.
                item['calls']=sum(role=='assistant' and not t.startswith('FINISH ') for role,t in s.h)
                out.append(dict(item,rescued=r,first=first_ok))
            e.save(name,out)
        print(f,shard,arm,len(out),flush=True)

def report():
    stats={};gate={};per={}
    for f in range(5):
        data={}
        for arm in ARMS:
            rows=[r for i in range(4) for r in e.read(f'part_{f}_{i}_{arm}')]
            data[arm]={r['id']:r for r in rows}
            assert len(rows)==len(data[arm])==84
        selected={}
        for k in range(3):
            opts=[]
            for arm in ('roles','rescue'):
                wins=loss=0
                for c in cases:
                    if c['k']!=k or folds[c['person']]!=(f+1)%5:continue
                    x=data['plain'][c['id']]['hit'];y=data[arm][c['id']]['hit']
                    wins+=y and not x;loss+=x and not y
                if wins>0 and loss==0:opts.append((arm,wins))
            selected[k]=max(opts,key=lambda o:o[1])[0] if opts else 'plain'
        gate[str(f)]=selected
        for c in cases:
            if folds[c['person']]!=f:continue
            rr={arm:dd[c['id']] for arm,dd in data.items()}
            rr['selected']=rr[selected[c['k']]]
            per[c['person']]=per.get(c['person'],0)+int(rr['selected']['hit'])-int(rr['plain']['hit'])
            for group in ('all',e.KINDS[c['k']],'numeric' if c['truth'] is not None else 'missing'):
                s=stats.setdefault(group,Counter());s['n']+=1
                for arm,r in rr.items():
                    for key in ('hit','raw','first','calls','rescued'):s[arm+'_'+key]+=r[key]
                for a,b in [('roles','before'),('before','after'),('rescue','remind'),('roles','plain'),('selected','plain')]:
                    x=rr[a]['hit'];y=rr[b]['hit'];k=a+'_vs_'+b
                    s[k+'_gain']+=x and not y;s[k+'_loss']+=y and not x;s[k+'_retained']+=x and y
    e.save('result',dict(stats=stats,gate=gate,patient_direction=dict(improved=sum(x>0 for x in per.values()),unchanged=sum(x==0 for x in per.values()),declined=sum(x<0 for x in per.values()))))

if __name__=='__main__':
    if sys.argv[1]=='report':
        report()
    else:
        v.setup_model()
        for fold in range(5):evaluate(fold,int(os.environ['PART']))
