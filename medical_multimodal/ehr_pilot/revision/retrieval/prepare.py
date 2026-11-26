import collections,csv,gzip,json,math,os,random,re
from datetime import datetime,timedelta
from pathlib import Path

def rows(name):
    with gzip.open(name+'.gz','rt') as f:yield from csv.DictReader(f)

def main(validation_start=24, mrn_base=49000100):
    stays={}
    for r in rows('p'):
        if not r['uniquepid'] or not r['unitdischargeoffset']:continue
        end=int(r['unitdischargeoffset'])
        if end<0:continue
        key=r['uniquepid'];sid=int(r['patientunitstayid'])
        if key not in stays or sid<stays[key][0]:stays[key]=(sid,min(1440,end))
    clocks=dict(stays.values()); measured=collections.defaultdict(dict)
    for r in rows('l'):
        sid=int(r['patientunitstayid'])
        if sid not in clocks or r['labname']!='potassium' or not all(r[x] for x in ('labresultoffset','labresultrevisedoffset','labresult')):continue
        if not {r[x].strip().lower() for x in ('labmeasurenamesystem','labmeasurenameinterface')}<={'mmol/l','meq/l'}:continue
        t,release=int(r['labresultoffset']),int(r['labresultrevisedoffset'])
        if max(t,release)>clocks[sid]:continue
        try:value=float(r['labresult'])
        except ValueError:continue
        if not math.isfinite(value):continue
        prev=measured[sid].get(t)
        if prev is None or prev[0]<release:measured[sid][t]=(release,{value})
        elif prev[0]==release:prev[1].add(value)
    groups=collections.defaultdict(list)
    for pid,(sid,clock) in stays.items():
        vals=measured[sid]
        if vals and len(vals[max(vals)][1])!=1:continue
        branch=('write' if next(iter(vals[max(vals)][1]))<3.5 else 'no_write') if vals else 'missing'
        groups[branch].append(pid)
    assert {k:len(ids) for k,ids in groups.items()}=={'write':203,'no_write':1488,'missing':148}
    rng=random.Random(26112709);dev=[];val=[];excluded=set()
    for branch in ('write','no_write','missing'):
        members=sorted(groups[branch]);rng.shuffle(members)
        dev+=members[:24]
        excluded.update(members[:validation_start])
        val.extend((p,branch) for p in members[validation_start:validation_start+24])
    rng.shuffle(val)
    assert len({pid for pid,_ in val}) == 72 and not excluded.intersection(pid for pid,_ in val)
    Path('reserved.json').write_text(json.dumps(dev))
    dataset=Path(os.environ['SOURCE'])/'artifacts/source/c/data/medagentbench/test_data_v2.json'
    template=next(r for r in json.loads(dataset.read_text()) if r['id'].startswith('task9_'))
    clock=datetime.fromisoformat('2023-11-13T10:15:00+00:00')
    cases=[];resources=[]
    for j,(pid,branch) in enumerate(val):
        sid,anchor=stays[pid];mrn='S'+str(mrn_base+j)
        measured_values=[]
        resources.append(dict(resourceType='Patient',id=mrn,identifier=[dict(value=mrn)]))
        for t,(_,vs) in sorted(measured[sid].items()):
            if len(vs)!=1:continue
            value=next(iter(vs));measured_values.append(value)
            resources.append(dict(resourceType='Observation',status='final',subject={'reference':'Patient/'+mrn},code={'coding':[{'code':'K'}]},effectiveDateTime=(clock+timedelta(minutes=t-anchor)).isoformat(),valueQuantity=dict(value=value,unit='mmol/L')))
        cases.append(dict(template,id='task9_'+str(j),eval_MRN=mrn,sol=None,
            instruction=re.sub(r'\bS\d+\b',mrn,template['instruction']),group=branch,source_values=measured_values))
    Path('cases.json').write_text(json.dumps(cases));Path('resources.json').write_text(json.dumps(resources))
if __name__=='__main__':main()
