"""Read-only event-log simulator, reference scoring, and patient partition."""
import datetime as dt
import json
import math
import random
import re
import tarfile
from pathlib import Path

KINDS=('breakfast_mean','week_change','meal_change')
CODES={'all':{48,57,58,59,60,61,62,63,64},'breakfast':{58},'supper':{62}}
TEXT=[
    'What is the mean pre-breakfast glucose during the seven calendar days ending on this date?',
    'How much did mean glucose across all readings change from the previous seven-day period to the latest seven-day period (latest minus previous)?',
    'Over the latest seven calendar days, what is the difference between mean pre-supper glucose and mean pre-breakfast glucose (supper minus breakfast)?']
GOALS=[
    [('breakfast','recent')],[('all','recent'),('all','previous')],[('supper','recent'),('breakfast','recent')]]
DOC="""You analyze a read-only diabetes event log. The current patient and anchor date are already scoped. Output exactly one action per turn. READ <group> <period> returns a handle and a count. Groups: all, breakfast (before breakfast), supper (before supper). Periods: recent (anchor day and previous six days), previous (the seven days before recent). STAT <handle> <operation> computes mean, min, max or count from its readings. DIFF <number> <number> returns first minus second. FINISH <number> returns a final answer. FINISH NA when any required group has no measurements. You can establish a missing group by reading it. Obtain a mean for every required nonempty group when reporting a numeric result. Tools return actual results. Never invent a handle. Eight actions maximum."""

def save(path,data):
    path=Path(path+'.json');tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data));tmp.replace(path)

def read(path):
    return json.loads(Path(path+'.json').read_text())

def prepare():
    patients={}
    with tarfile.open('raw.tar') as ar:
        for f in ar:
            if not re.fullmatch(r'Diabetes-Data/data-\d+',f.name) or not f.isfile():continue
            rows=[]
            for line in ar.extractfile(f).read().decode('latin1').splitlines():
                a=line.split()
                if len(a)!=4:continue
                try:
                    day=dt.datetime.strptime(a[0],'%m-%d-%Y').date().toordinal();code=int(a[2]);val=float(a[3])
                except (ValueError,OverflowError):continue
                if code in CODES['all'] and math.isfinite(val):rows.append([day,code,val])
            if rows:patients[f.name]=rows
    ids=sorted(patients);random.Random(26112708).shuffle(ids)
    cases=[];store={}
    for j,pid in enumerate(ids):
        rows=patients[pid];person=str(j);store[person]=rows
        stage='build' if j<len(ids)//2 else 'gate' if j<len(ids)//2+len(ids)//5 else 'test'
        days=sorted({r[0] for r in rows})
        for day in sorted({days[len(days)//2],days[-1]}):
            for k in range(3):
                # Reference calculation operates directly on rows and explicit date bounds.
                def reference_mean(codes,start,end):
                    xs=[v for t,c,v in rows if start<=t<=end and c in codes]
                    return sum(xs)/len(xs) if xs else None
                if k==0:
                    truth=reference_mean({58},day-6,day)
                elif k==1:
                    x=reference_mean(CODES['all'],day-6,day);y=reference_mean(CODES['all'],day-13,day-7)
                    truth=None if x is None or y is None else x-y
                else:
                    x=reference_mean({62},day-6,day);y=reference_mean({58},day-6,day)
                    truth=None if x is None or y is None else x-y
                cases.append(dict(id=str(len(cases)),person=person,stage=stage,k=k,day=day,truth=truth))
    save('patients',store);save('cases',cases)

class Session:
    def __init__(self,c,rows):
        self.c=c;self.rows=rows;self.bins={};self.seen={};self.means=set();self.h=[];self.done=False;self.value='invalid'
    def step(self,raw):
        self.h.append(['assistant',raw]);a=raw.split();response='Invalid action. Use a documented command.'
        if len(a)==3 and a[0]=='READ' and a[1] in CODES and a[2] in ('recent','previous'):
            d=self.c['day']-(7 if a[2]=='previous' else 0)
            xs=[v for t,c,v in self.rows if 0<=d-t<7 and c in CODES[a[1]]]
            handle='H'+str(len(self.bins)+1);key=tuple(a[1:])
            self.bins[handle]=(key,xs);self.seen[key]=len(xs)
            response=json.dumps(dict(handle=handle,count=len(xs)))
        elif len(a)==3 and a[0]=='STAT' and a[1] in self.bins and a[2] in ('mean','min','max','count'):
            key,xs=self.bins[a[1]]
            if a[2]=='mean':self.means.add(key)
            x=len(xs) if a[2]=='count' else None if not xs else {'mean':lambda:sum(xs)/len(xs),'min':lambda:min(xs),'max':lambda:max(xs)}[a[2]]()
            response=json.dumps(dict(value=x))
        elif len(a)==3 and a[0]=='DIFF':
            try:
                x=float(a[1])-float(a[2])
                if math.isfinite(x):response=json.dumps(dict(value=x))
            except ValueError:pass
        elif len(a)==2 and a[0]=='FINISH':
            self.done=True
            if a[1]=='NA':self.value=None
            else:
                try:self.value=float(a[1])
                except ValueError:pass
            return
        self.h.append(['user',response])
    def score(self):
        y=self.c['truth'];g=GOALS[self.c['k']]
        correct=self.done and ((self.value is None and y is None) or (isinstance(self.value,float) and y is not None and abs(y-self.value)<0.1))
        missing=any(self.seen.get(key)==0 for key in g)
        supported=missing if y is None else all(key in self.means for key in g)
        cause=('success' if correct and supported else 'missing required READ' if not (missing or all(key in self.seen for key in g)) else 'missing mean STAT' if not (missing or all(key in self.means for key in g)) else 'terminal value or format')
        return dict(id=self.c['id'],raw=bool(correct),hit=bool(correct and supported),cause=cause,calls=sum(role=='user' for role,_ in self.h),h=self.h)

def canonical(c,rows):
    s=Session(c,rows);means=[]
    for group,period in GOALS[c['k']]:
        s.step('READ '+group+' '+period)
        rec=json.loads(s.h[-1][1])
        if rec['count']==0:
            s.step('FINISH NA');break
        s.step('STAT '+rec['handle']+' mean')
        means.append(json.loads(s.h[-1][1])['value'])
    if not s.done:
        if len(means)==2:
            s.step('DIFF '+str(means[0])+' '+str(means[1]));x=json.loads(s.h[-1][1])['value']
        else:x=means[0]
        s.step('FINISH '+str(x))
    assert s.score()['hit']
    return '\n'.join(('Action: ' if role=='assistant' else 'Result: ')+t for role,t in s.h)

if __name__=='__main__':
    prepare()
    rows=read('patients');cases=read('cases')
    bank={c['id']:canonical(c,rows[c['person']]) for c in cases if c['stage']=='build'}
    save('bank',bank)
    print({k:(len({c['person'] for c in cases if c['stage']==k}),sum(c['stage']==k for c in cases)) for k in ('build','gate','test')},flush=True)
