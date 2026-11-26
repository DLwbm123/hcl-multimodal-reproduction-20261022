import datetime as dt
import json,os,sys
from collections import Counter
import compare as v
import engine as e

cases=e.read('cases');patients=e.read('patients');bank=e.read('bank')
train=[c for c in cases if c['stage']=='build']
def example(c):
    src=min((p for p in train if p['k']==c['k'] and p['person']!=c['person']),key=lambda p:abs(p['day']-c['day']))
    return bank[src['id']]

def evaluation(phase,shard):
    v.setup_model()
    for arm in (('plain','demo') if phase=='build' else ('plain','demo','score','credit')):
        subset=[c for c in cases if c['stage']==phase][shard::4]
        name=f'{phase}_{arm}_{shard}'
        from pathlib import Path
        out=e.read(name) if Path(name+'.json').exists() else []
        assert [x['id'] for x in out]==[c['id'] for c in subset[:len(out)]]
        for j in range(len(out),len(subset),6):
            batch=subset[j:j+6];sessions=[e.Session(c,patients[c['person']]) for c in batch]
            prompts=[]
            for c in batch:
                text=e.DOC+'\nAnchor date: '+dt.date.fromordinal(c['day']).isoformat()+'\nQuestion: '+e.TEXT[c['k']]
                if arm=='demo':text+='\nVerified example from a different patient. Use current receipts:\n'+example(c)
                if arm in ('score','credit'):text+='\nWorkflow memory:\n'+e.read('cards')[arm][c['k']]
                prompts.append(text)
            for _ in range(8):
                live=[i for i,s in enumerate(sessions) if not s.done]
                if not live:break
                chats=[[dict(role='user',content=prompts[i])]+[dict(role=r,content=t) for r,t in sessions[i].h] for i in live]
                for i,raw in zip(live,v.p.generate(chats,limit=64)):
                    sessions[i].step(raw.strip())
            out.extend(s.score() for s in sessions)
            e.save(name,out);print(name,len(out),'/',len(subset),flush=True)

def merged(phase,arm):
    rows=[r for i in range(4) for r in e.read(f'{phase}_{arm}_{i}')]
    result={r['id']:r for r in rows}
    assert len(rows)==len(result)==sum(c['stage']==phase for c in cases)
    return result

def edit():
    v.setup_model()
    plain,demo=merged('build','plain'),merged('build','demo');cards={}
    for arm in ('score','credit'):
        cards[arm]=[]
        for k in range(3):
            cs=[c for c in train if c['k']==k]
            # Both editors receive the same two frozen exemplars selected by outcome availability.
            chosen=[]
            for missing in (False,True):
                c=next((c for c in cs if (c['truth'] is None)==missing),None)
                if c:chosen.append(bank[c['id']])
            a=sum(plain[c['id']]['hit'] for c in cs);b=sum(demo[c['id']]['hit'] for c in cs)
            prompt='Draft one compact reusable workflow instruction for the question. State exact actions, binding to current receipts and when to stop. No example-specific answers. Output only the workflow.\n'+e.DOC+'\nQuestion '+e.TEXT[k]+'\nVerified examples:\n'+'\n'.join(chosen)+f'\nDevelopment successes without/with demo: {a}/{len(cs)} and {b}/{len(cs)}.'
            if arm=='credit':
                attr=Counter()
                for c in cs:
                    x,y=plain[c['id']],demo[c['id']]
                    attr[(x['cause'],y['cause'])]+=1
                prompt+='\nTransitions in first unmet evidence requirement, without -> with demo: '+str(dict(attr))
            cards[arm].append(v.p.generate([[dict(role='user',content=prompt)]],limit=256)[0].strip())
    e.save('cards',cards)

def report():
    gate={a:merged('gate',a) for a in ('demo','score','credit')}
    active={}
    for a in ('score','credit'):
        active[a]=[]
        for k in range(3):
            ids=[c['id'] for c in cases if c['stage']=='gate' and c['k']==k]
            wins=sum(gate[a][i]['hit'] and not gate['demo'][i]['hit'] for i in ids)
            losses=sum(gate['demo'][i]['hit'] and not gate[a][i]['hit'] for i in ids)
            active[a].append(dict(wins=wins,losses=losses,enabled=wins>0 and losses==0))
    arms={a:merged('test',a) for a in ('plain','demo','score','credit')};stats={}
    for c in [c for c in cases if c['stage']=='test']:
        k=c['k'];rows={a:data[c['id']] for a,data in arms.items()}
        for a in active:rows['selected_'+a]=rows[a] if active[a][k]['enabled'] else rows['demo']
        for group in ('all',e.KINDS[k],'missing' if c['truth'] is None else 'numeric'):
            s=stats.setdefault(group,Counter());s['n']+=1
            for a,r in rows.items():
                for label in ('hit','raw','calls'):s[a+'_'+label]+=r[label]
            for a,b in [('selected_credit','demo'),('selected_credit','selected_score'),('credit','score'),('credit','demo'),('demo','plain')]:
                x,y=rows[a]['hit'],rows[b]['hit'];prefix=a+'_vs_'+b+'_'
                s[prefix+'gain']+=x and not y;s[prefix+'loss']+=y and not x;s[prefix+'retained']+=x and y
    e.save('result',dict(gate=active,stats=stats))
if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='edit':edit()
    elif mode=='report':report()
    else:evaluation(mode,int(os.environ['PART']))
