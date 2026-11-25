"""Fit-only post-hoc probe of menu order and the brightness information supplied to memory."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from retina import MENU, SEED, SIZE, component, features, ratio, source, split, target, train
from shared import ROOT, guide, models, save

def main():
    fit_specialists, fit, _, _ = split()
    forest=train(fit_specialists)
    state=json.loads((Path(__import__('os').environ['PARENT_RUN'])/'private_update.json').read_text())
    choose,_=models()
    X=[]; errors=[]; choices=[]
    reversed_menu=("B = color expert using raw RGB and brightness; A = local-contrast expert using "
                   "green-channel contrast, local red-green differences and fine gradients. Reply with the expert letter.")
    for k in fit:
        im=source(k); ff,(x,y)=features(im)
        crop=Image.fromarray(im).crop((max(0,x-48),max(0,y-48),min(SIZE,x+48),min(SIZE,y+48)))
        xx=[round(np.asarray(crop).mean()),round(np.asarray(crop).std())]
        X.append(xx)
        q=f"Localized patch mean brightness {xx[0]}, standard deviation {xx[1]} on scale 0-255."
        visible={'whole':Image.fromarray(im),'patch':crop}
        hint=guide(state['skill'],state['memory'][0],q)
        observed=[]
        for rule in (SEED,hint):
            for menu,contrast_letter in [(MENU,'B'),(reversed_menu,'A')]:
                observed.append(int(choose(q,visible,rule+'\n'+menu,['A','B'])==contrast_letter))
        choices.append(observed)
        gt=target(k); truth=ratio(gt>0,gt==2); row=[]
        for i in range(2):
            f=ff[i]; local=f[:,:,2]<2.5; pred=np.zeros(local.shape,'uint8')
            pred[local]=forest[i].predict(f[local]);d=component(pred>0);c=component((pred==2)&d)
            row.append(abs(ratio(d,c)-truth))
        errors.append(row)
    X=np.array(X); e=np.array(errors); choices=np.array(choices)
    pred=np.empty(len(X),int); fixed=np.empty(len(X),int)
    for train_idx,val in KFold(n_splits=5,shuffle=True,random_state=20261125).split(X):
        model=DecisionTreeRegressor(max_depth=2,min_samples_leaf=8,random_state=7)
        model.fit(X[train_idx],e[train_idx,0]-e[train_idx,1])
        pred[val]=model.predict(X[val])>0
        fixed[val]=e[train_idx].mean(axis=0).argmin()
    idx=np.arange(len(X))
    save('result',dict(n=len(X),contrast_by_condition=choices.sum(axis=0).tolist(),
        order_changed_actions=[int((choices[:,i]!=choices[:,i+1]).sum()) for i in [0,2]],
        mae_by_condition=[e[idx,col].mean() for col in choices.T],
        cv_brightness_mae=e[idx,pred].mean(),cv_constant_mae=e[idx,fixed].mean(),
        cv_brightness_color=int((pred==0).sum()),oracle_mae=e.min(axis=1).mean()))
if __name__=='__main__': main()
