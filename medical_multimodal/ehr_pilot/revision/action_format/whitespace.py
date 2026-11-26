"""Post hoc replay with only POST-to-URL whitespace normalized."""
from collections import Counter,defaultdict
import json
from pathlib import Path
from urllib.parse import parse_qs,urlsplit
import compare as v

def read(name):
    return json.loads(Path(name+'.json').read_text())

cases={c['id']:c for c in read('cases')}
resources=defaultdict(list)
for resource in read('resources'):
    if resource['resourceType']=='Observation':
        resources[resource['subject']['reference']].append({'resource':resource})
def lookup(url):
    q=parse_qs(urlsplit(url).query)
    assert url.startswith(v.p.BASE+'Observation?') and q['code']==['K']
    return {'status_code':200,'data':json.dumps({'entry':resources['Patient/'+q['patient'][0]]})}
v.p.ref=v.p.scorer()
v.p.ref.send_get_request=lookup
counts=defaultdict(Counter)
for i in range(3):
    for row in read(f'constrained_{i}'):
        case=cases[row['id']]
        old=row['history']
        fixed=[(role,'POST '+s[5:].lstrip() if role=='agent' and s.startswith('POST ') else s) for role,s in old]
        for name,history in [('observed',old),('normalized',fixed)]:
            answer=history[-1][1][7:-1] if row['status']=='finish' else None
            reads=[('Observation',parse_qs(urlsplit(old[0][1][4:]).query))]
            official,qualified=v.p.grade(case,history,answer,reads)
            if name=='observed':
                assert (official,qualified)==(row['official'],row['qualified'])
            counts[case['group']][name]+=qualified
print(json.dumps(counts,indent=2))
