import json
from collections import Counter
from pathlib import Path
from itertools import product
import sys

def read(p):
    return json.loads(p.read_text())

def delta(a,b,la,lb):
    if a==b: return (0,0)
    return (min(x-y for x,y in product(la,lb)),max(x-y for x,y in product(la,lb)))

def summarize(root):
    rows={}; ratings=[{},{}]; reviews={}
    for p in sorted(root.glob('part[0-5]')):
        assert read(p/'SUPERVISOR_EXIT.json')['success']
        rows.update(read(p/'private_generated.json'))
        for d,n in zip(ratings,('primary','second')):d.update(read(p/('private_'+n+'.json')))
    for p in sorted(root.glob('audit[0-1]')):
        assert read(p/'SUPERVISOR_EXIT.json')['success']
        reviews.update(read(p/'private_review.json'))
    assert len(rows)==200 and all(r.keys()==rows.keys() for r in ratings)
    result={'n':len(rows),'trigger':sum(r['trigger'] for r in rows.values()),'scores':{},'costs':{},'evidence_review':{}}
    for arm in 'SKGVD':
        stat=[0.]*4; tools=0
        for r in rows.values():
            for i,v in enumerate(r['costs'][arm]):stat[i]+=v
            if arm in 'KGV':
                for i,v in enumerate(r['costs']['S']):stat[i]+=v
            tools+=1 if arm=='D' else len(r['notes'])+int(r['trigger'] and arm in 'GV')
        result['costs'][arm]=dict(tools=tools,requests=int(stat[0]),input_tokens=int(stat[1]),output_tokens=int(stat[2]),seconds=round(stat[3],2))
    for subset in ['all','trigger']:
        keys=[k for k,r in rows.items() if subset=='all' or r['trigger']]
        group={}
        for mode in ['primary','second','agreement']:
            absolute={b:Counter() for b in 'SKGVD'};pairs={p:Counter() for p in ('VS','KS','GS','DS','VD','VG','GD','KV')}
            for k in keys:
                ans=rows[k]['answers']
                def score(b):
                    v=[d[k][ans[b]] for d in ratings]
                    return v[0] if mode=='primary' or (mode=='agreement' and v[0]==v[1]) else v[1] if mode=='second' else 'U'
                for b in absolute:absolute[b][score(b)]+=1
                for p,c in pairs.items():
                    a,b=p
                    poss={'1':(1,),'0':(0,),'U':(0,1)}
                    low,high=delta(ans[a],ans[b],poss[score(a)],poss[score(b)])
                    c['gain']+=low==1;c['loss']+=high==-1;c['unresolved']+=low!=high;c['net_low']+=low;c['net_high']+=high
            group[mode]=dict(absolute=absolute,paired=pairs)
        result['scores'][subset]=group
    result['reviewer_disagreement_answers']=sum(sum(v!=ratings[1][k][a] for a,v in lab.items()) for k,lab in ratings[0].items())
    result['unique_scored_answers']=sum(map(len,ratings[0].values()))
    result['evidence_review']['reviewed_cases']=len(reviews)
    expected={k for k,r in rows.items() if (int(k)-1)%10==0 or any(ratings[0][k][a]!=ratings[1][k][a] for a in r['answers'].values()) or any(len(set(j[k].values()))>1 for j in ratings)}
    assert reviews.keys()==expected
    for arm in 'SKGVD':
        for field in ['support','extra']:
            result['evidence_review'][arm+'_'+field]=dict(Counter(v[arm][field] for v in reviews.values()))
    result['trigger_prefix']=dict(Counter(str(len(r['notes'])) for r in rows.values() if r['trigger']))
    result['target_none']=sum(r['trigger'] and (not r['cue'] or r['cue'].strip().rstrip('.').upper()=='NONE') for r in rows.values())
    result['first_tool_all']=Counter(r['notes'][0]['name'] for r in rows.values())
    profile={}
    for k,r in rows.items():
        if not r['trigger']:continue
        note=r['notes'][0]; c=profile.setdefault(note['name'],Counter())
        c['n']+=1
        c['success_with_text']+=bool(note['ok'] and note['text'].strip())
        for arm,answer in r['answers'].items():
            c['correct_'+arm]+=ratings[0][k][answer]=='1'
    result['trigger_by_first_tool']=profile
    result['trigger_strata']={}
    for state in ('empty','nonempty'):
        selected=[k for k,r in rows.items() if ('nonempty' if r['trusted'] else 'empty')==state]
        result['trigger_strata'][state]=dict(n=len(selected),at_limit=sum(len(rows[k]['notes'])==4 for k in selected))
    overlap=Counter()
    support=Counter()
    for k,r in rows.items():
        if not r['trigger']:continue
        a=r['answers']; score=lambda arm:ratings[0][k][a[arm]]
        overlap[''.join(score(b) for b in 'KGV')]+=1
        if score('S')!=score('V'):
            key=score('S')+'_'+score('V')
            support[key+'_'+reviews[k]['V']['support']]+=1
    result['trigger_joint_KGV']=overlap
    result['primary_flip_visual_support']=support
    # Include physically executed worker costs separately from the five standalone policies.
    physical=Counter()
    for p in sorted(list(root.glob('part[0-5]'))+list(root.glob('audit[0-1]'))):
        for line in (p/'tokens.jsonl').read_text().splitlines():
            name,inp,out=json.loads(line)
            physical[name]+=1;physical['input_tokens']+=inp;physical['output_tokens']+=out
    result['actual_including_review']=physical
    return result

if __name__=='__main__':print(json.dumps(summarize(Path(sys.argv[1])),indent=2))
