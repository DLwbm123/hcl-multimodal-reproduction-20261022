import collections
import datetime as dt
import json
import os
from pathlib import Path
import original as o

v,p,d=o.v,o.p,o.d


def cases():
    parts,records=o.source()
    old={r['id']:r for j in range(4) for r in o.read('previous/part'+str(j))}
    bank=o.read('archive/library')
    rules=o.read('previous/rules')
    for i in parts['valid']:
        if old[i]['arms']['selected']['hit']:
            continue
        case=v.ROWS[i];obs=o.observation(case,records[i]);item=v.match(case,bank)
        route=rules[str(item['key'])]
        plan=o.predict(route['rule'],obs['features']) if route['enabled'] else tuple(route['rule']['mode'])
        yield case,obs,item['card'],plan,old[i]['arms']['selected']


def capture(case,obs,card,plan,truth,ans):
    raw=[]
    generate=p.generate
    def logged(*args,**kwargs):
        result=generate(*args,**kwargs)
        raw.extend(result)
        return result
    p.generate=logged
    try:
        metrics=o.simulate(case,obs,card,plan,truth,ans)
    finally:
        p.generate=generate
    return dict(metrics=metrics,raw=raw)


def numeric_audit(obs,ans,raw):
    last=raw[-1].strip()
    try:
        value=json.loads(last[7:-1]) if last.startswith('FINISH(') and last.endswith(')') else None
    except ValueError:
        value=None
    points=sorted((dt.datetime.fromisoformat(r['effectiveDateTime']),r['valueQuantity']['value']) for r in obs['rows'])
    window=[x for t,x in points if d.MONITOR-dt.timedelta(days=1)<=t<=d.MONITOR]
    hist=[x for _,x in points]
    candidates={'missing':-1}
    if hist:candidates.update(history_mean=sum(hist)/len(hist),latest=hist[-1],oldest=hist[0])
    if window:candidates.update(window_mean=sum(window)/len(window),window_last=window[-1],window_first=window[0])
    scalar=value[0] if isinstance(value,list) and len(value)==1 and type(value[0]) in (int,float) else None
    return dict(model=value,reference=ans,window_values=window,source=points and [[t.isoformat(),x] for t,x in points],
                matches=[k for k,x in candidates.items() if scalar is not None and abs(scalar-x)<0.1],
                form='invalid' if value is None else 'list_'+str(len(value)) if isinstance(value,list) else type(value).__name__)


def run():
    v.setup_model()
    shard=int(os.environ['PART'])
    rows=[]
    for case,obs,card,plan,old in list(cases())[shard::4]:
        truth,ans=o.reference(case,obs)
        result=capture(case,obs,card,plan,truth,ans)
        rows.append(dict(id=case['id'],write=len(truth)>1,plan_ok=plan==truth,same=old==result['metrics'],
                         audit=numeric_audit(obs,ans,result['raw']),**result))
        p.save('r'+str(shard)+'.json',rows)
        print(len(rows),flush=True)


def check():
    rows=[r for j in range(4) for r in o.read('r'+str(j))]
    assert len(rows)==len({r['id'] for r in rows})==21
    groups=collections.defaultdict(collections.Counter)
    for r in rows:
        g=groups[r['id'].split('_')[0]]
        g.update(n=1,reproduced=r['same'],plan_correct=r['plan_ok'],write=r['write'])
        g['form_'+r['audit']['form']]+=1
        for k in r['audit']['matches'] or ['none']:g['matches_'+k]+=1
    p.save('replay.json',groups)
    assert all(r['same'] for r in rows)

if __name__=='__main__':
    import sys
    (check if sys.argv[1]=='check' else run)()
