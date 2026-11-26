import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor, export_text
from adapt import append
from eye import BASE, bundle, gold, outputs, probability, report, score, shape_score
from retina import SIZE, features, ratio, split, train
from previous_learning import update_memory
from shared import ROOT, example, models, save

NAMES=['cup_confidence','disc_solidity','cup_solidity','first_vcdr']
GROUPS={'B':[0],'C':[1,2,3],'D':[0,1,2,3]}
CUT=.002


def observe(key,partition,forest,*,paired):
    with Image.open(BASE/partition/'images'/(key+'.jpg')) as src:
        image=np.array(src.convert('RGB').resize((SIZE,SIZE),Image.Resampling.LANCZOS))
    (color,contrast),_=features(image)
    p=probability(forest[1],contrast)
    disc,cup=outputs(p)
    grad=cv2.magnitude(cv2.Sobel(image[:,:,1],cv2.CV_32F,1,0),cv2.Sobel(image[:,:,1],cv2.CV_32F,0,1))
    x=[float(p[:,:,2][cup].mean()) if cup.any() else 0.,shape_score(disc,grad)[0],shape_score(cup,grad)[0],ratio(disc,cup)]
    if not paired:
        return x
    truth=gold(key,partition)
    label=int(key.startswith('g'))
    first=score((disc,cup),truth,label)
    second=score(outputs((p+probability(forest[0],color))*.5),truth,label)
    return x,first['error']-second['error']


def tree(x,y,indices):
    return DecisionTreeRegressor(max_depth=2,min_samples_leaf=max(4,int(np.ceil(.15*len(y)))),random_state=12).fit(x[:,indices],y)


def fit_policy(tag,keys,observations,choose,gen):
    x=np.array([observations[k][0] for k in keys])
    y=np.array([observations[k][1] for k in keys])
    votes={'A':np.zeros(len(y),bool)}
    rules={'A':'stop after first segmentation'}
    trees={}
    for code,cols in GROUPS.items():
        votes[code]=np.zeros(len(y),bool)
        for a,b in KFold(n_splits=5,shuffle=True,random_state=16).split(x):
            votes[code][b]=tree(x[a],y[a],cols).predict(x[b][:,cols])>CUT
        t=tree(x,y,cols)
        trees[code]=t
        rules[code]=export_text(t,feature_names=[NAMES[i] for i in cols],decimals=4)+'; review when value > 0.002'
    credits={}
    for keep,drop in [('B','C'),('C','B')]:
        credits[keep]=(y>CUT)&votes['D']&votes[keep]&~votes[drop]
    examples=[]
    for i,k in enumerate(keys):
        record=dict(first=dict(zip(NAMES,x[i])),error_reduction=y[i],
                    credited=[name for name,v in credits.items() if v[i]],oof_review=[name for name,v in votes.items() if v[i]])
        examples.append(example(k,json.dumps(record),'review' if votes['D'][i] else 'stop','review' if y[i]>CUT else 'stop'))
    path=ROOT/('memory_'+tag)
    path.mkdir()
    memory=update_memory({'memories':[]},tag,examples,path,gen,capacity=len(keys))[0]
    scores={code:dict(net_error_reduction=float((y*mask).mean()),requests=int(mask.sum()),
                      improved=int(((y>0)&mask).sum()),worsened=int(((y<0)&mask).sum()))
            for code,mask in votes.items()}
    info=dict(out_of_fold=scores,credit={k:int(v.sum()) for k,v in credits.items()},
              abstract=memory['abstract'],rules=rules)
    letter=choose('Fit-only evidence: '+json.dumps(info),{},
                  'Select a deployable review rule. Prioritize highest out-of-fold mean error reduction; if tied prefer fewer requests. A=stop, B=cup confidence, C=shape, D=combined. Return one letter.',list('ABCD'))
    skill=append('Apply the fitted rule and request a second segmentation only when it votes to review.',letter)
    selected=skill.split()[-1]
    assert selected==letter
    stats=dict(n=len(keys),selected=selected,scores=scores,credit=info['credit'],beneficial=int((y>CUT).sum()),
               harmful=int((y< -CUT).sum()),rule=rules[selected])
    def act(x):
        if selected=='A':
            return np.zeros(len(x),bool)
        return trees[selected].predict(x[:,GROUPS[selected]])>CUT
    return act,stats


def main():
    tool,fit,gate,prior=split()
    forest=train(tool)
    choose,gen=models()
    observations={k:observe(k,'train',forest,paired=True) for k in fit}
    groups={}
    for seed in (31,32,33):
        rng=np.random.default_rng(seed)
        keys=[]
        for prefix,n in [('g',3),('n',27)]:
            keys.extend(rng.choice([k for k in fit if k.startswith(prefix)],n,replace=False).tolist())
        rng.shuffle(keys)
        groups['n30_'+str(seed)]=keys
    groups['n80']=fit
    policies={}
    result={'fit':{}}
    for tag,ids in groups.items():
        policies[tag],result['fit'][tag]=fit_policy(tag,ids,observations,choose,gen)
        print(tag,result['fit'][tag]['selected'],flush=True)
    save('result',result)
    earlier=Path(os.environ['PRIOR_RUN'])
    assert json.loads((earlier/'SUPERVISOR_EXIT.json').read_text())['success']
    for phase,partition,keys in [('gate','train',gate),('old_classification','train',prior),
              ('new_measurement','val',sorted(p.stem for p in (BASE/'val/images').glob('*.jpg')))]:
        x=np.array([observe(k,partition,forest,paired=False) for k in keys])
        choices={tag:act(x) for tag,act in policies.items()}
        # Freeze all decisions before reading the previously scored potential outcomes.
        cached=json.loads((earlier/('private_'+phase+'.json')).read_text())
        assert set(keys)==set(cached)
        base=[cached[k][0] for k in keys]
        assert np.allclose([r['v'] for r in base],x[:,-1],rtol=0,atol=1e-12)
        arms={'stop':base,'previous':[cached[k][3] for k in keys]}
        for tag,flags in choices.items():
            arms[tag]=[cached[k][int(flag)] for k,flag in zip(keys,flags)]
        summary={}
        for tag,rows in arms.items():
            summary[tag]=dict(metrics=report(rows),vs_stop=bundle(base,rows))
        summary['80_vs_30']=bundle(arms['n30_31'],arms['n80'])
        summary['80_vs_previous']=bundle(arms['previous'],arms['n80'])
        if phase=='gate':
            result['gate_passed']={tag:summary[tag]['vs_stop']['delta_MAE']<0 and summary[tag]['metrics']['bacc']>=summary['stop']['metrics']['bacc'] for tag in policies}
        result[phase]=summary
        save('result',result)
        print(phase,len(keys),flush=True)


if __name__=='__main__':
    main()
