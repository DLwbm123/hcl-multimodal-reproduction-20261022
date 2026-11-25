import os
import random
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier
from shared import calls, example, guide, learn, models, save

cv2.setNumThreads(1)
DATA = Path(os.environ["FUNDUS_ROOT"]) / "train"
SIZE = 320
SEED = "Choose the segmentation expert expected to give the most accurate vertical cup-to-disc ratio on this image."
MENU = ("A = color expert using raw RGB and brightness; B = local-contrast expert using "
        "green-channel contrast, local red-green differences and fine gradients. Reply with the expert letter.")


def split():
    rng = random.Random(20261125)
    groups = [[] for _ in range(4)]
    for prefix, counts in [('g',(12,8,4,16)),('n',(108,72,36,144))]:
        ids = sorted(p.stem for p in (DATA / 'images').glob(prefix + '*.jpg'))
        assert len(ids) == sum(counts)
        rng.shuffle(ids)
        i = 0
        for group,n in zip(groups,counts):
            group.extend(ids[i:i+n])
            i += n
    for group in groups:
        rng.shuffle(group)
    return groups


def source(k):
    with Image.open(DATA / 'images' / (k + '.jpg')) as im:
        return np.asarray(im.convert('RGB').resize((SIZE,SIZE), Image.Resampling.LANCZOS))


def target(k):
    with Image.open(DATA / 'mask' / (k + '.bmp')) as im:
        m = np.asarray(im.convert('L').resize((SIZE,SIZE), Image.Resampling.NEAREST))
    return (m < 200).astype('uint8') + (m < 60).astype('uint8')


def features(im):
    a = im.astype('float32')/255
    r,g,b = a.transpose(2,0,1)
    y,x = np.mgrid[:SIZE,:SIZE]
    # A broad brightness maximum within the retinal field defines a label-free inspection window.
    score = cv2.GaussianBlur(.35*r+.65*g,(0,0),8)
    score[(r<.12)|(y<40)|(y>SIZE-40)|(x<32)|(x>SIZE-32)] = -1
    cy,cx = np.unravel_index(score.argmax(),score.shape)
    dx,dy = (x-cx)/32,(y-cy)/32
    distance = np.hypot(dx,dy)
    common = [dx,dy,distance]
    bg = cv2.GaussianBlur(g,(0,0),10)
    color = np.stack(common+[r,g,b,score], axis=-1).astype('float32')
    local = np.stack(common+[g,g-bg,r-g,g-b,cv2.GaussianBlur(g,(0,0),2),abs(cv2.Laplacian(g,cv2.CV_32F))],axis=-1).astype('float32')
    return (color,local),(int(cx),int(cy))


def train(ids):
    rng=np.random.default_rng(20261125)
    X=[[],[]]; Y=[]
    for k in ids:
        im=source(k); pair,_=features(im); mask=target(k).ravel()
        parts=[np.flatnonzero((mask==0)&(pair[0][:,:,2].ravel()<2.5)),np.flatnonzero(mask==1),np.flatnonzero(mask==2)]
        indices=np.concatenate([rng.choice(p,min(len(p),400),replace=False) for p in parts])
        for buf,feat in zip(X,pair):buf.append(feat.reshape(-1,feat.shape[-1])[indices])
        Y.append(mask[indices])
    return [ExtraTreesClassifier(n_estimators=40,n_jobs=2,min_samples_leaf=4,random_state=14).fit(np.concatenate(buf),np.concatenate(Y)) for buf in X]


def component(m):
    n,l,s,_=cv2.connectedComponentsWithStats(m.astype('uint8'))
    return l==(1+s[1:,cv2.CC_STAT_AREA].argmax()) if n>1 else m.astype(bool)


def height(m):
    yy=np.flatnonzero(np.any(m,axis=1))
    return int(yy[-1]-yy[0]+1) if len(yy) else 0


def ratio(disc,cup):
    return height(cup)/max(1,height(disc))


def dice(a,b):
    return 2*int((a&b).sum())/max(1,int(a.sum()+b.sum()))


def balanced(rows):
    return sum(sum(v["prediction"]==v["label"] for v in rows if v["label"]==y) /
               sum(v["label"]==y for v in rows) for y in (0,1))/2


def summarize(rows):
    return dict(n=len(rows),correct=sum(x["prediction"]==x["label"] for x in rows),balanced_accuracy=balanced(rows),
                vCDR_MAE=float(np.mean([v["error"] for v in rows])),
                disc_dice=float(np.mean([v["disc"] for v in rows])),cup_dice=float(np.mean([v["cup"] for v in rows])),
                contrast_calls=sum(v["route"]==1 for v in rows))


def main():
    toolfit,fit,gate,test=split()
    forest=train(toolfit)
    choose,gen=models()

    @lru_cache(maxsize=16)
    def prepare(k):
        im=source(k); ff,(cx,cy)=features(im)
        crop=Image.fromarray(im).crop((max(0,cx-48),max(0,cy-48),min(SIZE,cx+48),min(SIZE,cy+48)))
        q=f"Localized patch mean brightness {np.asarray(crop).mean():.0f}, standard deviation {np.asarray(crop).std():.0f} on scale 0-255."
        return ff,{'whole':Image.fromarray(im),'patch':crop},q

    @lru_cache(None)
    def execute(k,j):
        f=prepare(k)[0][j]; local=f[:,:,2]<2.5
        mask=np.zeros(local.shape,'uint8');mask[local]=forest[j].predict(f[local])
        d=component(mask>0);c=component((mask==2)&d)
        return d,c,ratio(d,c)

    def select(k,skill=SEED,memory=None):
        _,imgs,q=prepare(k)
        hint=guide(skill,memory,q) if memory else skill
        return int(choose(q,imgs,hint+'\n'+MENU,['A','B'])=='B')

    def score(k,j,threshold):
        d,c,v=execute(k,j); t=target(k); td=t>0;tc=t==2
        return dict(route=j,vcdr=v,error=abs(v-ratio(td,tc)),disc=dice(d,td),cup=dice(c,tc),
                    label=int(k.startswith('g')),prediction=int(v>=threshold))
    samples=[]; fitrows=[]; baseline={}
    for k in fit:
        j=select(k); baseline[k]=j
        pair=[score(k,i,.5) for i in range(2)]; fitrows.append(pair)
        best=min(range(2),key=lambda i:pair[i]['error'])
        samples.append(example(k,prepare(k)[2],['color','contrast'][j],['color','contrast'][best]))
    threshold=max(np.arange(.15,.951,.025),key=lambda t:balanced([dict(v,prediction=int(v['vcdr']>=t)) for pair in fitrows for v in pair]))
    fixed=min(range(2),key=lambda i:sum(pair[i]['error'] for pair in fitrows))
    print('fit complete',flush=True)
    skill,memory=learn(SEED,samples,gen)
    result={"threshold":float(threshold),"fixed_expert":fixed,"valid_edit":bool(skill)}
    accepted=False
    for name,ids in [('gate',gate),('test',test)]:
        rows={}
        for i,k in enumerate(ids):
            b=select(k)
            candidate=select(k,skill,memory) if skill and (name=='gate' or accepted) else b
            rows[k]=[score(k,j,threshold) for j in (fixed,b,candidate)]
            if (i+1)%40==0: print(name,i+1,flush=True)
        arms=[[v[col] for v in rows.values()] for col in range(3)]
        f,b,c=map(summarize,arms)
        if name=='gate':
            accepted=bool(skill and c['vCDR_MAE']<b['vCDR_MAE'] and c['balanced_accuracy']>=b['balanced_accuracy'])
        result[name]=dict(fixed=f,baseline=b,learned=c,changed_routes=sum(p['route']!=q['route'] for p,q in zip(arms[1],arms[2])),
                          gain=sum(p['prediction']!=p['label'] and q['prediction']==q['label'] for p,q in zip(arms[1],arms[2])),
                          loss=sum(p['prediction']==p['label'] and q['prediction']!=q['label'] for p,q in zip(arms[1],arms[2])))
        save('private_'+name,rows)
        result.update(accepted=accepted,calls=dict(calls));save('progress',result)
    save('result',result)
