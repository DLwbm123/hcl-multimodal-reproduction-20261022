"""Construct the score-blind external panel from PhysioNet demo v2.2."""
import collections
import csv
from datetime import datetime, timedelta, timezone, date
import gzip
import json
import math
import os
from pathlib import Path
import random
import re

clock = datetime(2023, 11, 13, 10, 15)
fields = {'50960': ('MG', 'mg/dL'), '50971': ('K', 'mEq/L'),
          '52610': ('K', 'mEq/L'), '50852': ('A1C', '%'), '225664': ('GLU', '')}


def rows(file):
    with gzip.open(file + '.gz', 'rt') as stream:
        yield from csv.DictReader(stream)


def build():
    patients = {r['subject_id']: r for r in rows('p')}
    assert len(patients) == 100
    anchors = {}
    for f, start, end, hours in [('a','admittime','dischtime',48), ('u','intime','outtime',24)]:
        for row in sorted(rows(f), key=lambda r:r[start], reverse=True):
            begin, finish = [datetime.fromisoformat(row[k]) for k in (start,end)]
            anchors[row['subject_id']] = min(begin + timedelta(hours=hours), finish)
    records = collections.defaultdict(dict)
    for f in ('l', 'c'):
        for r in rows(f):
            if r['itemid'] not in fields or not r['valuenum'] or not r['storetime']: continue
            code,unit = fields[r['itemid']]
            if r['valueuom'] != unit: continue
            sid = r['subject_id']
            ts, store = (datetime.fromisoformat(r[k]) for k in ('charttime','storetime'))
            anchor = anchors[sid]
            val = float(r['valuenum'])
            if max(ts,store)>anchor or ts<anchor-timedelta(days=1825) or not math.isfinite(val): continue
            key = code, ts
            rec = (store, int(r.get('labevent_id') or 0), val)
            if key in records[sid] and records[sid][key][:2] == rec[:2] and records[sid][key][2] != val:
                raise ValueError('Conflicting values at identical release time')
            records[sid][key] = max(records[sid].get(key,rec),rec)
    root = Path(os.environ['SOURCE'])/'artifacts/source/c/data/medagentbench'
    templates = {}
    for row in json.loads((root/'test_data_v2.json').read_text()):
        if row.get('eval_MRN'): templates.setdefault(row['id'].split('_')[0],row)
    names = [(a,b) for a in 'Ada Alex Brian Carol David Emma Helen James Julia Mark'.split()
             for b in 'Baker Brooks Cole Cooper Foster Green Hall Lane Reed Stone'.split()]
    rng = random.Random(261128)
    rng.shuffle(names)
    numbers=rng.sample(range(1000000,10000000),100)
    bank=json.loads(Path('library.json').read_text())
    historical_text=json.dumps(templates)
    cases=[]; resources=[]; strata=collections.defaultdict(collections.Counter)
    for idx,sid in enumerate(sorted(patients)):
        mrn='S'+str(numbers[idx])
        assert mrn not in historical_text
        anchor=anchors[sid]
        age=int(patients[sid]['anchor_age'])+anchor.year-int(patients[sid]['anchor_year'])
        doy=date(2001,1,1)+timedelta(days=rng.randrange(365))
        dob=date(clock.year-age-int((doy.month,doy.day)>(clock.month,clock.day)),doy.month,doy.day)
        first,last=names[idx]
        resources.append(dict(resourceType='Patient',id=mrn,identifier=[{'value':mrn}],
                              birthDate=dob.isoformat(),name=[dict(given=[first],family=last)]))
        values=collections.defaultdict(list)
        for (code,ts),(_,_,v) in sorted(records[sid].items(),reverse=True):
            mapped=(clock+(ts-anchor)).replace(tzinfo=timezone.utc).isoformat()
            unit=dict(MG='mg/dL',K='mmol/L',A1C='%',GLU='mg/dL')[code]
            observation=dict(resourceType='Observation',status='final',subject={'reference':'Patient/'+mrn},
                             code={'coding':[{'code':code}]},effectiveDateTime=mapped,
                             valueQuantity=dict(value=v,unit=unit))
            resources.append(observation)
            values[code].append((ts,v))
        for n in range(1,11):
            kind='task'+str(n)
            case=templates[kind].copy()
            if n==1:
                case['instruction']=re.sub(r'name .+? and DOB of [\d-]+',f'name {first} {last} and DOB of {dob}',case['instruction'])
                case['sol']=[mrn]
            else:
                case['instruction']=re.sub(r'\bS\d+\b',mrn,case['instruction'])
                case['sol']=None
            case.update(id=f'{kind}_{idx:03}',eval_MRN=mrn)
            group='standard'
            if n in (4,5,6,7,9,10):
                code={4:'MG',5:'MG',6:'GLU',7:'GLU',9:'K',10:'A1C'}[n]
                seq=[(t,v) for t,v in values[code] if n not in (4,5,6) or t>=anchor-timedelta(days=1)]
                if not seq: group='missing'
                elif n==5: group='write' if seq[0][1]<=1.9 else 'no_write'
                elif n==9: group='write' if seq[0][1]<3.5 else 'no_write'
                elif n==10: group='old_write' if (anchor-seq[0][0])>timedelta(days=365) else 'recent_no_write'
                else: group='measured'
            strata[kind][group]+=1
            case['stratum']=group
            cases.append(case)
    rng.shuffle(cases)
    Path('cases.json').write_text(json.dumps(cases))
    Path('resources.json').write_text(json.dumps(resources))
    Path('coverage.json').write_text(json.dumps(dict(n=len(cases),patients=len(patients),strata=strata,
                                                    observations=len(resources)-len(patients)),indent=2))
    print(Path('coverage.json').read_text())


if __name__=='__main__':
    build()
