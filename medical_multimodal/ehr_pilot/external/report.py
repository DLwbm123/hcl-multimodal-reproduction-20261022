"""Report paired outcomes and cluster uncertainty with each external patient held together."""
import collections
import json
import os
from pathlib import Path
import random
import compare as v

load=lambda name:json.loads(Path(name+'.json').read_text())
cases=load('cases')
bank=load('library')
gates={'plain':None,'demo':load('demo_gate')['enabled'],'card':load('gate')['enabled']}
arms={}
for arm,gate in gates.items():
    records=[]
    for i in range(int(os.environ['SHARDS'])):
        expected=[c['id'] for c in cases[i::int(os.environ['SHARDS'])] if gate is None or v.match(c,bank)['key'] in gate]
        chunk=load(f'{arm}_{i}') if expected else []
        assert [r['id'] for r in chunk]==expected
        records.extend(chunk)
    assert len({r['id'] for r in records})==len(records)
    arms[arm]=dict(arms.get('plain',{}))
    arms[arm].update({r['id']:r for r in records})
    assert len(arms[arm])==1000


def summarize(subset):
    ids={c['id'] for c in subset}
    hits={a:{i for i in ids if rows[i]['qualified']} for a,rows in arms.items()}
    card,demo,plain=(hits[k] for k in ('card','demo','plain'))
    return dict(n=len(ids),plain=len(plain),demo=len(demo),card=len(card),gain=len(card-demo),
                loss=len(demo-card),both=len(card&demo),neither=len(ids-(card|demo)),
                demo_retained=len(demo&plain),card_retained=len(card&plain))


def costs(rows):
    return dict(steps=sum(x['steps'] for x in rows.values()),tools=sum(
        role=='agent' and msg.startswith(('GET ','POST ')) for x in rows.values() for role,msg in x['history']))

by=collections.defaultdict(list)
for c in cases: by[c['eval_MRN']].append(c)
vals=[]
for rows in by.values():
    t=summarize(rows)
    vals.append((t['gain']-t['loss'])/t['n'])
rng=random.Random(261128)
samples=sorted(sum(rng.choices(vals,k=len(vals)))/len(vals) for _ in range(10000))
groups=sorted({(v.d.kind(c),c['stratum']) for c in cases})
result=dict(overall=summarize(cases),patients=len(by),rate_diff_ci95=[samples[249],samples[9749]],
    lab=summarize([c for c in cases if v.d.kind(c) in v.d.CODES]),
    by_workflow={n:summarize([c for c in cases if v.d.kind(c)==n]) for n in range(1,11)},
    by_branch={f'{n}_{s}':summarize([c for c in cases if v.d.kind(c)==n and c['stratum']==s]) for n,s in groups},
    costs={a:costs(r) for a,r in arms.items()})
v.p.save('result.json',result)
