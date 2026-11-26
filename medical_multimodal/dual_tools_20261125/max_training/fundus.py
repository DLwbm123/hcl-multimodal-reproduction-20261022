"""Maximum feedback with independent measurement and historical diagnosis panels."""
import json
import os
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold
import hcl.memory.raw as native_raw
from eye import BASE, bundle, report
from feature_eye import fit_policy, observe
from retina import split, train
from shared import models, save


def diverse_take(candidates, count, already):
    """Same diversity ordering as RawMemory, updating nearest distances incrementally."""
    if count<=0 or not candidates:
        return []
    entries=list(candidates)
    tokens=[native_raw._question_tokens(r) for r in entries]
    ids=[str(native_raw._source(r,'task_id')) for r in entries]
    indices=list(range(len(entries)))
    closest=np.ones(len(entries))
    def update(t):
        for i in indices:
            closest[i]=min(closest[i],1.0-native_raw._token_jaccard(tokens[i],t))
    for r in already:
        update(native_raw._question_tokens(r))
    result=[]
    while indices and len(result)<count:
        best=indices[-1] if not already and not result else max(indices,key=lambda i:(closest[i],ids[i]))
        indices.remove(best)
        result.append(entries[best])
        if indices and len(result)<count:
            update(tokens[best])
    return result


def panel(keys,partition,forest,policies):
    rows={tag:[] for tag in ['stop','always',*policies]}
    for k in keys:
        x,a,b=observe(k,partition,forest,paired=True,return_scores=True)
        # Only the first-pass features enter the decision. Cached potential outcomes
        # permit paired comparisons for the different rules on this same image.
        base=dict(a,review=0)
        review=dict(b,review=1)
        rows['stop'].append(base)
        rows['always'].append(review)
        for tag,act in policies.items():
            rows[tag].append(review if act(np.array([x]))[0] else base)
    result={tag:dict(metrics=report(rr),vs_stop=bundle(rows['stop'],rr)) for tag,rr in rows.items()}
    result['large_vs_small']=bundle(rows['n80'],rows['max'])
    return result


def main():
    tool,small,gate,old=split()
    v=sorted(p.stem for p in (BASE/'val/images').glob('*.jpg'))
    test=sorted(p.stem for p in (BASE/'test/images').glob('*.jpg'))
    all_keys=tool+small+gate+old+v
    assert len(all_keys)==len(set(all_keys))==800 and len(test)==400 and set(test).isdisjoint(all_keys)
    for part,ids in [('train',tool+small+gate+old),('val',v),('test',test)]:
        assert set(ids)=={p.stem for p in (BASE/part/'mask').iterdir()}
    sample=[dict(question=s,internal_source={'task_id':str(i)}) for i,s in enumerate(['a b','a','a c','','a b','b c'])]
    for start in ([],sample[:1]):
        for n in range(7):
            assert native_raw._diverse_take(sample[1:],n,start)==diverse_take(sample[1:],n,start)
    native_raw._diverse_take=diverse_take
    forest=train(tool)
    obs={}
    for a,b in KFold(5,shuffle=True,random_state=16).split(tool):
        partial=train([tool[j] for j in a])
        for j in b:
            obs[tool[j]]=observe(tool[j],'train',partial,paired=True)
    for part,ids in [('train',small+gate+old),('val',v)]:
        for i,k in enumerate(ids):
            obs[k]=observe(k,part,forest,paired=True)
            if i%100==99:print(part,i+1,flush=True)
    choose,gen=models()
    policies={}
    result={'fit':{}}
    for tag,keys in [('n80',small),('n640',tool+small+gate+v),('n800',all_keys)]:
        policies[tag],result['fit'][tag]=fit_policy(tag,keys,obs,choose,gen)
        print(tag,result['fit'][tag]['selected'],flush=True)
        save('result',result)
    result['measurement']=panel(test,'test',forest,{'n80':policies['n80'],'max':policies['n800']})
    save('result',result)
    result['diagnosis']=panel(old,'train',forest,{'n80':policies['n80'],'max':policies['n640']})
    prior=json.loads((Path(os.environ['EYE_PRIOR'])/'result.json').read_text())['old_classification']['n80']['metrics']
    control=result['diagnosis']['n80']['metrics']
    result['reproduced_80_diagnosis']=all(abs(prior[key]-control[key])<1e-10 for key in prior)
    save('result',result)


if __name__=='__main__':
    main()
