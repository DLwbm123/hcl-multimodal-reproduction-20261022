"""Check the six nonmatching-value cases using the already frozen replay outputs."""
import json
from collections import Counter
from pathlib import Path

def read(name):
    return json.loads(Path(name+'.json').read_text())

def diffs(a,b,path=''):
    if isinstance(a,dict) and isinstance(b,dict) and a.keys()==b.keys():
        return sum((diffs(a[k],b[k],path+'.'+k) for k in a),[])
    if isinstance(a,list) and isinstance(b,list) and len(a)==len(b):
        return sum((diffs(x,y,path+'[]') for x,y in zip(a,b)),[])
    return [path] if a!=b else []

from prior import latest,render
from t import nearest,rebound
cases={c['id']:c for c in read('cases') if c['stage']=='test'}
bank,policy=read('bank'),read('policy')
counts=Counter()
for i in range(3):
    for row in read(f'old_{i}'):
        c=cases[row['id']]
        if c['group']!='write':
            continue
        x=latest(json.loads(row['history'][1][1])['series'])
        ex=nearest(bank,x)
        if ex['latest']==x:
            continue
        retrieved=rebound(ex,c['eval_MRN'])
        fitted=render(policy['template'],c['eval_MRN'],x)
        different=diffs(retrieved,fitted)
        counts['missing_write' if not retrieved else ','.join(different)]+=1
print(json.dumps(counts,indent=2))
