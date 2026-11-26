import collections,csv,gzip,json,math,os,random,re
from datetime import datetime,timedelta
from pathlib import Path

def rows(name):
    with gzip.open(name+'.gz','rt') as f:yield from csv.DictReader(f)

def candidates(lab):
    is_mg=lab=='magnesium'
    units={'mg/dl'} if is_mg else {'mmol/l','meq/l'}
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
        if sid not in clocks or r['labname']!=lab or not all(r[x] for x in ('labresultoffset','labresultrevisedoffset','labresult')):continue
        if not {r[x].strip().lower() for x in ('labmeasurenamesystem','labmeasurenameinterface')}<=units:continue
        t,release=int(r['labresultoffset']),int(r['labresultrevisedoffset'])
        if max(t,release)>clocks[sid] or (is_mg and t<clocks[sid]-1440):continue
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
        branch='missing'
        if vals:
            x=next(iter(vals[max(vals)][1]))
            write=x<=1.9 if is_mg else x<3.5
            branch='write' if write else 'no_write'
        groups[branch].append(pid)
    return stays,measured,groups

def main():
    _,_,prior=candidates('potassium')
    assert {k:len(v) for k,v in prior.items()}=={'write':203,'no_write':1488,'missing':148}
    rng=random.Random(26112709)
    exclude=set()
    phases=('write','no_write','missing')
    for group in phases:
        ids=sorted(prior[group]);rng.shuffle(ids);exclude.update(ids[:120])
    assert len(exclude)==360
    stays,measured,groups=candidates('magnesium')
    rng=random.Random(26112705)
    val=[]
    for group in phases:
        ids=sorted(set(groups[group])-exclude)
        rng.shuffle(ids)
        assert len(ids)>=264,(group,len(ids))
        for i,pid in enumerate(ids[:264]):
            val.append((pid,group,'build' if i<120 else 'gate' if i<144 else 'test'))
    assert len(set(x[0] for x in val))==792 and not exclude.intersection(x[0] for x in val)
    rng.shuffle(val)
    dataset=Path(os.environ['SOURCE'])/'artifacts/source/c/data/medagentbench/test_data_v2.json'
    template=next(r for r in json.loads(dataset.read_text()) if r['id'].startswith('task5_'))
    clock=datetime.fromisoformat('2023-11-13T10:15:00+00:00')
    cases=[];resources=[]
    for j,(pid,branch,stage) in enumerate(val):
        sid,anchor=stays[pid];mrn='S'+str(49400100+j)
        measured_values=[]
        resources.append(dict(resourceType='Patient',id=mrn,identifier=[dict(value=mrn)]))
        for t,(_,vs) in sorted(measured[sid].items()):
            if len(vs)!=1:continue
            value=next(iter(vs));measured_values.append(value)
            resources.append(dict(resourceType='Observation',status='final',subject={'reference':'Patient/'+mrn},code={'coding':[{'code':'MG'}]},effectiveDateTime=(clock+timedelta(minutes=t-anchor)).isoformat(),valueQuantity=dict(value=value,unit='mg/dL')))
        cases.append(dict(template,id='task5_'+str(j),eval_MRN=mrn,sol=None,
            instruction=re.sub(r'\bS\d+\b',mrn,template['instruction']),group=branch,stage=stage,source_values=measured_values))
    Path('cases.json').write_text(json.dumps(cases));Path('resources.json').write_text(json.dumps(resources))
if __name__=='__main__':main()
