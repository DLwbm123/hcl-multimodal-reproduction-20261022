"""Evaluate the already frozen candidate after binding its unambiguous replace to the only editable slot."""
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from patch_format import normalize
from upstream_skill import apply_patch_with_report
from retina import SIZE, SEED, MENU, component, dice, features, ratio, source, target, train, split, summarize
from shared import guide, models, save

def main():
    old=Path(os.environ['PARENT_RUN'])
    proposal=json.loads((old/'private_update.json').read_text())
    raw=proposal['attempts'][0]['output'].strip().removeprefix('```json').removesuffix('```').strip()
    obj=json.loads(raw)
    edit,=obj['edits']
    assert set(edit)=={'op','content'} and edit['op']=='replace' and len(edit['content'])<=700
    edit['target']=SEED
    patch,_=normalize(json.dumps(obj),SEED)
    skill,receipt=apply_patch_with_report(SEED,patch)
    assert receipt[0]['status']=='applied_replace'
    toolsplit,_,gate,test=split()
    forest=train(toolsplit)
    choose,_=models(); memory=proposal['memory'][0]
    threshold=json.loads((old/'progress.json').read_text())['threshold']
    accepted=False; result={}
    for phase,ids in [('gate',gate),('test',test)]:
        base=json.loads((old/('private_'+phase+'.json')).read_text())
        assert set(ids)==set(base)
        pairs=[]
        for k in ids:
            previous=base[k][1]
            if phase=='test' and not accepted:
                pairs.append((previous,previous)); continue
            im=source(k); ff,(x,y)=features(im)
            crop=Image.fromarray(im).crop((max(0,x-48),max(0,y-48),min(SIZE,x+48),min(SIZE,y+48)))
            arr=np.asarray(crop); q=f"Localized patch mean brightness {arr.mean():.0f}, standard deviation {arr.std():.0f} on scale 0-255."
            j=int(choose(q,{'whole':Image.fromarray(im),'patch':crop},guide(skill,memory,q)+'\n'+MENU,['A','B'])=='B')
            f=ff[j]; local=f[:,:,2]<2.5; mask=np.zeros(local.shape,'uint8'); mask[local]=forest[j].predict(f[local])
            d=component(mask>0); c=component((mask==2)&d); value=ratio(d,c); gt=target(k)
            candidate=dict(route=j,vcdr=value,error=abs(value-ratio(gt>0,gt==2)),disc=dice(d,gt>0),cup=dice(c,gt==2),
                           label=previous['label'],prediction=int(value>=threshold))
            pairs.append((previous,candidate))
        b,c=map(summarize,zip(*pairs))
        if phase=='gate': accepted=c['vCDR_MAE']<b['vCDR_MAE'] and c['balanced_accuracy']>=b['balanced_accuracy']
        result[phase]=dict(baseline=b,learned=c,changed_routes=sum(i['route']!=j['route'] for i,j in pairs))
        result['accepted']=accepted
        save('private_'+phase,dict(zip(ids,pairs)));save('result',result)
if __name__=='__main__': main()
