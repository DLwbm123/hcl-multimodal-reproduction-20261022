import json, math, os, random, re
from collections import defaultdict
from datetime import datetime,timedelta
from pathlib import Path
import baseprep as base

def main():
    stays,_,pot=base.candidates('potassium')
    _,_,mag=base.candidates('magnesium')
    excluded=set()
    for groups,seed,size in ((pot,26112709,120),(mag,26112705,312)):
        rng=random.Random(seed)
        earlier=set(excluded)
        for key in ('write','no_write','missing'):
            ids=sorted(set(groups[key])-earlier);rng.shuffle(ids)
            excluded.update(ids[:size])
    assert len(excluded)==1296
    eligible={pid:s for pid,s in stays.items() if pid not in excluded}
    clocks=dict(eligible.values())
    series=defaultdict(dict)
    for r in base.rows('l'):
        sid=int(r['patientunitstayid'])
        if sid not in clocks or r['labname']!='bedside glucose':continue
        if any(r[k].strip().lower()!='mg/dl' for k in ('labmeasurenamesystem','labmeasurenameinterface')):continue
        try:
            t,release,x=int(r['labresultoffset']),int(r['labresultrevisedoffset']),float(r['labresult'])
        except ValueError:continue
        if max(t,release)>clocks[sid] or not math.isfinite(x):continue
        old=series[sid].get(t)
        if old is None or old[0]<release:series[sid][t]=(release,{x})
        elif old[0]==release:old[1].add(x)
    bins=defaultdict(list)
    for pid,(sid,anchor) in eligible.items():
        if any(len(v)!=1 for _,v in series[sid].values()):continue
        n=sum(t>=anchor-1440 for t in series[sid])
        bins['empty' if n==0 else 'one' if n==1 else 'multiple'].append(pid)
    rng=random.Random(26112707)
    dataset=json.loads((Path(os.environ['SOURCE'])/'artifacts/source/c/data/medagentbench/test_data_v2.json').read_text())
    templates={f:next(c for c in dataset if c['id'].startswith(f'task{f}_')) for f in (6,7)}
    clock=datetime.fromisoformat('2023-11-13T10:15:00+00:00');cases=[];resources=[]; counts={}
    for group in ('empty','one','multiple'):
        ids=sorted(bins[group]);rng.shuffle(ids)
        a,b=len(ids)//2,int(len(ids)*0.15)
        counts[group]=[a,b,len(ids)-a-b]
        for j,pid in enumerate(ids):
            sid,anchor=eligible[pid];mrn='S'+str(49600100+len(cases)//2)
            resources.append(dict(resourceType='Patient',id=mrn,identifier=[dict(value=mrn)]))
            points=[]
            for t,(_,vs) in sorted(series[sid].items()):
                x=next(iter(vs));time=(clock+timedelta(minutes=t-anchor)).isoformat()
                points.append([time,x])
                resources.append(dict(resourceType='Observation',status='final',subject={'reference':'Patient/'+mrn},code={'coding':[{'code':'GLU'}]},effectiveDateTime=time,valueQuantity=dict(value=x,unit='mg/dL')))
            for f,c in templates.items():
                cases.append(dict(c,id=f'task{f}_{mrn}',eval_MRN=mrn,sol=None,instruction=re.sub(r'\bS\d+\b',mrn,c['instruction']),points=points,group=group,stage='build' if j<a else 'gate' if j<a+b else 'test'))
    for name,obj in [('cases',cases),('resources',resources),('counts',counts)]:
        Path(name+'.json').write_text(json.dumps(obj))
if __name__=='__main__':main()
