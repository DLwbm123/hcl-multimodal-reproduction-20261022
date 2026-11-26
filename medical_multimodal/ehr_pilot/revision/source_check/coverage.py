"""Score-blind coverage under fixed stay, release-time, and unit criteria."""
from collections import Counter,defaultdict
import csv
import gzip
import json
import math
from pathlib import Path
import sys

def run(directory):
    def read(name):
        with gzip.open(directory/(name+'.gz'),'rt') as f:yield from csv.DictReader(f)
    stays={}
    for r in read('p'):
        if not r['uniquepid'] or not r['unitdischargeoffset']:continue
        end=int(r['unitdischargeoffset'])
        if end<0:continue
        pid=r['uniquepid']; stay=int(r['patientunitstayid'])
        if pid not in stays or stay<stays[pid][0]:stays[pid]=(stay,min(1440,end))
    anchors=dict(stays.values());last={};conflicts=set()
    for r in read('l'):
        code=r['labname'];stay=int(r['patientunitstayid'])
        if code not in ('magnesium','potassium') or stay not in anchors:continue
        if not r['labresultrevisedoffset'] or not r['labresult'] or not r['labresultoffset']:continue
        units={r[k].strip().lower() for k in ('labmeasurenamesystem','labmeasurenameinterface')}
        valid={'mg/dl'} if code=='magnesium' else {'mmol/l','meq/l'}
        if not units<=valid:continue
        t,release=int(r['labresultoffset']),int(r['labresultrevisedoffset'])
        if max(t,release)>anchors[stay] or (code=='magnesium' and t<anchors[stay]-1440):continue
        try:v=float(r['labresult'])
        except ValueError:continue
        if not math.isfinite(v):continue
        key=(stay,code);stamp=(t,release)
        if key not in last or stamp>last[key][0]:
            last[key]=(stamp,v);conflicts.discard(key)
        elif stamp==last[key][0] and v!=last[key][1]:conflicts.add(key)
    result={}
    for code in ('magnesium','potassium'):
        n=Counter()
        for stay in anchors:
            key=(stay,code)
            if key in conflicts:n['ambiguous']+=1
            elif key not in last:n['missing']+=1
            else:
                v=last[key][1]
                need=v<=1.9 if code=='magnesium' else v<3.5
                n['write' if need else 'no_write']+=1
        result[code]=dict(n)
    return dict(patients=len(stays),counts=result)
if __name__=='__main__':print(json.dumps(run(Path(sys.argv[1])),indent=2))
