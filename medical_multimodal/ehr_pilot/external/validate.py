"""Check the installed FHIR records and official grading against the frozen source fixtures."""
from pathlib import Path
import collections
from datetime import datetime, timedelta, date
import json
import compare as v

cases=json.loads(Path('cases.json').read_text())
resources=json.loads(Path('resources.json').read_text())
patients={r['id']:r for r in resources if r['resourceType']=='Patient'}
series=collections.defaultdict(list)
for r in resources:
    if r['resourceType']=='Observation':
        s=r['subject']['reference'].split('/')[-1]
        series[s,r['code']['coding'][0]['code']].append((datetime.fromisoformat(r['effectiveDateTime']),r['valueQuantity']['value']))
v.p.ref=v.p.scorer()
hist_source={r.get('eval_MRN') for r in v.ROWS.values()}
assert not patients.keys() & hist_source
assert len(cases)==len({c['id'] for c in cases})==1000
counts=collections.Counter()
for case in cases:
    n=v.d.kind(case)
    mrn=case['eval_MRN']
    ok,rows,posts,answer=v.d.canonical(case,return_trace=True)
    expected=[]
    if n==1:expected=[mrn]
    elif n==2:
        dob=date.fromisoformat(patients[mrn]['birthDate'])
        today=v.d.MONITOR.date()
        expected=[today.year-dob.year-int((today.month,today.day)<(dob.month,dob.day))]
    elif n in v.d.CODES:
        seq=sorted(series[mrn,v.d.CODES[n]],reverse=True)
        assert len(rows)==len(seq)
        if n in (4,5,6): seq=[q for q in seq if q[0]>=v.d.MONITOR-timedelta(days=1)]
        if n in (4,7):expected=[seq[0][1] if seq else -1]
        if n==6:expected=[sum(t[1] for t in seq)/len(seq) if seq else -1]
        if n==10:
            expected=[seq[0][1],seq[0][0].isoformat()] if seq else [-1]
            if len(answer)==2: answer=[answer[0],datetime.fromisoformat(answer[1]).isoformat()]
    needs_write=case['stratum'] in ('write','old_write') or n in (3,8) or (n==10 and case['stratum']=='missing')
    assert ok and answer==expected and bool(posts)==needs_write, (n,'environment control failed')
    counts[n]+=1
assert set(counts.values())=={100}
v.p.save('controls.json',dict(checked=len(cases),by_workflow=counts))
