"""Verify pairings and aggregate provenance without exporting patient records."""
import collections
import json
import os
import random
from pathlib import Path
from urllib.parse import urlsplit,parse_qs


def load(path):
    return json.loads(Path(path).read_text())


def main():
    split=load('partition.json')
    source=Path(os.environ['SOURCE'])/'artifacts/source/c/data/medagentbench/test_data_v2.json'
    cases={c['id']:c for c in load(source)}
    patients={k:{cases[i]['eval_MRN'] for i in ids} for k,ids in split.items()}
    assert not (patients['history'] & (patients['train'] | patients['valid']) or patients['train'] & patients['valid'])
    history=split['history']
    library=load('library.json')
    assert all(x in split['train'] for item in library for x in item['sources'])
    from compare import match, d
    enabled=load('gate.json')['enabled']
    arms={}
    for name in ['plain','demo','card']:
        items=[r for path in Path('.').glob('history_'+name+'_*.json') for r in load(path)]
        assert len(items)==len(history)==len(set(r['id'] for r in items))
        arms[name]={r['id']:r for r in items}
    arms['selected']={i:(arms['card'][i] if match(cases[i],library)['key'] in enabled else arms['plain'][i]) for i in history}
    scored=load('result.json')
    rows={}
    historical=[r['id'] for r in load(Path(os.environ['PRIOR'])/'baseline.json') if r['qualified']]
    for name,arm in arms.items():
        by=collections.defaultdict(collections.Counter)
        evidence_count=0
        for i in history:
            b=bool(arms['plain'][i]['qualified'])
            q=bool(arm[i]['qualified'])
            if q and d.kind(cases[i]) not in d.WRITES:
                expected=urlsplit(d.query(cases[i]))
                wanted=parse_qs(expected.query)
                h=arm[i]['history']
                for (role,t),(_,reply) in zip(h,h[1:]):
                    if role!='agent' or not t.startswith('GET ') or not reply.startswith('{"returned":'):
                        continue
                    url=urlsplit(t[4:])
                    if url.path==expected.path and all(set(v)<=set(parse_qs(url.query).get(k,[])) for k,v in wanted.items()):
                        evidence_count+=1
                        break
            elif q:
                evidence_count+=1
            by[i.split('_')[0]].update(n=1,correct=int(q),gain=int(q and not b),loss=int(b and not q),kept=int(b and q))
        actual=scored['arms'][name]
        assert sum(v['correct'] for v in by.values())==actual['correct']
        assert sum(v['gain'] for v in by.values())==actual['gains']
        assert sum(v['loss'] for v in by.values())==actual['losses']
        rows[name]=dict(by_family=by,correct_with_canonical_read_or_write=evidence_count,historical_retained=sum(arm[i]['qualified'] for i in historical))
    # Exploratory patient-cluster resampling for the head-to-head rate difference.
    grouped=collections.defaultdict(list)
    for i in history:
        grouped[cases[i]['eval_MRN']].append(int(arms['selected'][i]['qualified'])-int(arms['demo'][i]['qualified']))
    groups=list(grouped.values()); rng=random.Random(261127); samples=[]
    for _ in range(10000):
        sampled=rng.choices(groups,k=len(groups))
        samples.append(sum(map(sum,sampled))/sum(map(len,sampled)))
    samples.sort()
    out=dict(patients={k:len(v) for k,v in patients.items()},teacher_source_counts={str(i['key']):len(i['sources']) for i in library},rows=rows,selected_minus_demo_patient_bootstrap_95=[samples[249],samples[9749]])
    Path('pairs.json').write_text(json.dumps(out,indent=2)+'\n')


if __name__=='__main__':
    main()
