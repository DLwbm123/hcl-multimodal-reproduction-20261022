import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor, export_text
from adapt import append, consolidate
from retina import SIZE, component, dice, features, height, ratio, split, train
from shared import calls, example, models, save

BASE=Path(os.environ['FUNDUS_ROOT'])
FIELDS=['first_vcdr','disc_solidity','cup_solidity','disc_retained','cup_retained',
        'cup_confidence','cup_edge_strength','disc_edge_strength','visual_quality_grade']
POLICIES=[(1,12,0.002),(2,12,0.002),(2,20,0.002),(2,12,0.006)]
CUTOFF=.5  # Threshold calibrated in the prior run using only its 80 feedback images.


def outputs(prob):
    pred=prob.argmax(axis=2)
    disc=component(pred>0)
    cup=component((pred==2)&disc)
    return disc,cup


def probability(model,feat):
    local=feat[:,:,2]<2.5
    p=np.zeros((*local.shape,3),np.float32)
    p[:,:,0]=1.
    p[local]=model.predict_proba(feat[local])
    return p


def outline(mask):
    return cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]


def shape_score(mask,grad):
    contours=outline(mask)
    if not contours:return 0.,0.
    contour=max(contours,key=cv2.contourArea)
    edge=contour.reshape(-1,2)
    area=cv2.contourArea(contour)
    return area/max(1.,cv2.contourArea(cv2.convexHull(contour))),float(grad[edge[:,1],edge[:,0]].mean())


def first_state(im,ff,ctr,model,choose):
    p=probability(model,ff); disc,cup=outputs(p)
    grad=cv2.magnitude(cv2.Sobel(im[:,:,1],cv2.CV_32F,1,0),cv2.Sobel(im[:,:,1],cv2.CV_32F,0,1))/1020.
    ds,de=shape_score(disc,grad);cs,ce=shape_score(cup,grad)
    overlay=im.copy()
    cv2.drawContours(overlay,outline(disc),-1,(0,255,80),1)
    cv2.drawContours(overlay,outline(cup),-1,(255,60,200),1)
    cx,cy=ctr
    box=(max(0,cx-50),max(0,cy-50),min(SIZE,cx+51),min(SIZE,cy+51))
    seen={'raw':Image.fromarray(im).crop(box),'outline':Image.fromarray(overlay).crop(box)}
    letter=choose('Same optic nerve head twice. Green line = outer disc, pink = cup.',seen,
                  'Do these boundaries follow the visible disc and cup? A=convincing, B=uncertain, C=clearly misplaced or missing. Return one letter.',list('ABC'))
    hard=p.argmax(axis=2)
    state=np.array([ratio(disc,cup),ds,cs,disc.sum()/max(1,(hard>0).sum()),cup.sum()/max(1,(hard==2).sum()),
                    p[:,:,2][cup].mean() if cup.any() else 0.,ce,de,'ABC'.index(letter)],float)
    return state,p,(disc,cup)


def gold(k,partition):
    suffix='.png' if partition=='val' else '.bmp'
    with Image.open(BASE/partition/'mask'/(k+suffix)) as src:
        mask=np.asarray(src.convert('L').resize((SIZE,SIZE),Image.Resampling.NEAREST))
    assert set(np.unique(mask)).issubset({0,128,255})
    return mask<200,mask<60


def score(pair,truth,label):
    disc,cup=pair;d,c=truth
    v=ratio(disc,cup)
    return dict(error=abs(v-ratio(d,c)),disc=dice(disc,d),cup=dice(cup,c),v=v,label=label,pred=int(v>=CUTOFF))


def report(rows):
    s={k:float(np.mean([r[k] for r in rows])) for k in ['error','disc','cup']}
    s['reviews']=sum(r['review'] for r in rows)
    s['n']=len(rows)
    if rows[0]['label'] is not None:
        s['correct']=sum(r['label']==r['pred'] for r in rows)
        s['sensitivity']=float(np.mean([r['pred']==1 for r in rows if r['label']==1]))
        s['specificity']=float(np.mean([r['pred']==0 for r in rows if r['label']==0]))
        s['bacc']=(s['sensitivity']+s['specificity'])/2
    return s


def bundle(base,other):
    delta=np.array([b['error']-a['error'] for a,b in zip(base,other)])
    rng=np.random.default_rng(120)
    result=dict(delta_MAE=float(delta.mean()),ci95=np.quantile(delta[rng.integers(0,len(delta),(3000,len(delta)))].mean(axis=1),[.025,.975]).tolist(),
                error_better=int(sum(delta<0)),error_worse=int(sum(delta>0)))
    if base[0]['label'] is not None:
        result.update(gain=sum(a['pred']!=a['label'] and b['pred']==b['label'] for a,b in zip(base,other)),
                      loss=sum(a['pred']==a['label'] and b['pred']!=b['label'] for a,b in zip(base,other)))
    return result


def main():
    toolfit,fit,gate,old_test=split()
    forest=train(toolfit)
    choose,gen=models()
    def process(k,partition,decide):
        with Image.open(BASE/partition/'images'/(k+'.jpg')) as src:
            im=np.array(src.convert('RGB').resize((SIZE,SIZE),Image.Resampling.LANCZOS))
        (color,contrast),ctr=features(im)
        x,p,first=first_state(im,contrast,ctr,forest[1],choose)
        flags=decide(x)
        # The unconditional fusion comparator consumes the second expert for every image.
        # No policy that decides to stop sees that computation.
        second=outputs((p+probability(forest[0],color))*.5) if any(flags) else first
        t=gold(k,partition)
        label=int(k.startswith('g')) if partition=='train' else None
        a=score(first,t,label); b=score(second,t,label)
        return x,[dict(b if v else a,review=int(v)) for v in flags]

    X=[]; y=[]; examples=[]
    for i,k in enumerate(fit):
        x,(a,b)=process(k,'train',lambda x:[False,True])
        benefit=a['error']-b['error']
        X.append(x);y.append(benefit)
        desc='First-mask evidence: '+', '.join(f'{n}={v:.3f}' for n,v in zip(FIELDS,x))
        ex=example(k,desc+f'. Paired measurement errors: keep={a["error"]:.4f}, review={b["error"]:.4f}.',
                   'keep','review' if benefit>.002 else 'keep')
        examples.append(ex)
    X=np.array(X); y=np.array(y); memory=consolidate(examples,gen)
    rules=[]; models_fit=[]
    for depth,leaf,cut in POLICIES:
        oof=np.zeros(len(y),bool)
        for a,b in KFold(n_splits=5,shuffle=True,random_state=16).split(X):
            m=DecisionTreeRegressor(max_depth=depth,min_samples_leaf=leaf,random_state=12).fit(X[a],y[a])
            oof[b]=m.predict(X[b])>cut
        m=DecisionTreeRegressor(max_depth=depth,min_samples_leaf=leaf,random_state=12).fit(X,y)
        rules.append(dict(tree=export_text(m,feature_names=FIELDS,decimals=4),cut=cut,
                          oof_error_reduction=float((y*oof).mean()),oof_reviews=int(oof.sum())))
        models_fit.append(m)
    prompt=('Select a single deployable rule for reviewing a first predicted cup/disc contour using a second expert. '
            'Favor greater cross-validated absolute error reduction; if tied prefer fewer reviews. '
            'Trees predict the reduction; review only above their cutoff. Frozen HCL fit memories: '
            +json.dumps(memory['abstract'])+'\n'+json.dumps(dict(zip('ABCD',rules))))
    selected=choose(prompt,{},'Return one rule letter.',list('ABCD'))
    available=dict(zip('ABCD',rules))
    edit=append('Inspect initial contour quality.',selected)
    key=edit.split()[-1]
    if key not in available:raise ValueError('unrecognized executable skill')
    j='ABCD'.index(key)
    save('private_rule',dict(edit=edit,selected=key,rules=rules,memory=memory))
    def decide(x):
        baseline=x[-1]==2 or x[5]<.55
        learned=float(models_fit[j].predict(x.reshape(1,-1))[0])>POLICIES[j][2]
        return [False,True,baseline,learned]
    fresh=sorted(p.stem for p in (BASE/'val/images').glob('*.jpg'))
    assert len(fresh)==400 and set(fresh).isdisjoint(toolfit+fit+gate+old_test)
    result=dict(fit_n=len(fit),selected=key,fit_oof=rules[j]['oof_error_reduction'])
    accepted=False
    for phase,partition,ids in [('gate','train',gate),('old_classification','train',old_test),('new_measurement','val',fresh)]:
        rows={}
        for i,k in enumerate(ids,1):
            x,r=process(k,partition,decide)
            rows[k]=r
            if i%50==0:print(phase,i,flush=True)
        arms=list(zip(*rows.values()))
        summaries=list(map(report,arms)); base,cand=summaries[-2:]
        if phase=='gate':accepted=cand['error']<base['error'] and cand['bacc']>=base['bacc']
        result[phase]=dict(zip(['always_keep','always_review','no_learning','learned'],summaries))
        result[phase].update(paired=bundle(arms[-2],arms[-1]),deployed=cand if accepted else base)
        result.update(accepted=accepted,model_calls=dict(calls));save('private_'+phase,rows);save('result',result)
