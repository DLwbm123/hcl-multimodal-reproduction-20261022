"""Paired model-vs-model audit; emits aggregate counts only."""
import json
from collections import Counter
from pathlib import Path

def load(name):
    return json.loads(Path(name+'.json').read_text())

def arm(name):
    return {row['id']:row['qualified'] for part in range(4) for row in load(name+str(part))}

a,b=arm('base'),arm('demo')
out={}
for c in load('cases'):
    if c['stage']!='test':continue
    x,y=a[c['id']],b[c['id']]
    for group in ('all',c['id'].split('_')[0]):
        out.setdefault(group,Counter()).update(n=1,gain=int(y and not x),loss=int(x and not y),retained=int(x and y))
print(json.dumps(out,indent=2))
