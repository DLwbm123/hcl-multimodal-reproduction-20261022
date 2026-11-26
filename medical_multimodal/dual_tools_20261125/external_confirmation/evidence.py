"""Blinded ancillary review of disagreements, flips, and a fixed concordance sample."""
import gc
import json
import os
import random
import time
from pathlib import Path
from PIL import Image
from acquire import load_vlm
from observation import region_view, LOCAL
from shared import ROOT


def read(p):
    return json.loads(p.read_text())


def run():
    parent=Path(os.environ['PANEL_ROOT'])
    inputs={x['id']:x for x in read(parent/'inputs.json')}
    labels=read(parent/'labels.json')
    os.environ['MODEL_ROOT']=os.environ['REFEREE1']
    choose=load_vlm()
    output={}
    for shard in range(int(os.environ['PART']),6,2):
        source=parent/('part'+str(shard))
        while not (source/'SUPERVISOR_EXIT.json').exists():time.sleep(3)
        assert read(source/'SUPERVISOR_EXIT.json')['success']
        primary=read(source/'private_primary.json'); second=read(source/'private_second.json')
        for k,r in read(source/'private_generated.json').items():
            a=r['answers']
            disputed=any(primary[k][v]!=second[k][v] for v in a.values())
            flip=any(len(set(g[k][v] for v in a.values()))>1 for g in (primary,second))
            if not (flip or disputed or (int(k)-1)%10==0):continue
            q=inputs[k]['q']; img=Image.open(inputs[k]['path']).convert('RGB')
            notes={'S':r['trusted'],'D':[r['whole']]}
            if r['trigger']:
                notes.update(K=[n for n in r['notes'] if n['ok']],G=[r['whole']],V=[r['local']])
            else:
                notes.update({b:r['trusted'] for b in 'KGV'})
            unique={}
            for b,obs in notes.items():
                key=(a[b],json.dumps(obs,sort_keys=True))
                unique.setdefault(key,[]).append(b)
            order=list(unique)
            random.Random(125+int(k)).shuffle(order)
            output[k]={}
            for ans,obs_text in order:
                obs=json.loads(obs_text)
                images={'source':img}
                for i,n in enumerate(obs):
                    if n and n['name']==LOCAL and n['ok']:
                        images[str(i)+'_view']=region_view(img,json.loads(n['arg'])['box'])
                ctx='Question: '+q+'\nReference: '+labels[k]+'\nAnswer: '+ans+'\nObservations supplied to answerer: '+json.dumps([n['text'][:450] for n in obs if n])
                support=choose(ctx,images,'Does the actual supplied observation identify a relevant visible fact supported by the displayed image and sufficient to justify the reference answer? A yes; B missing, irrelevant, or contradicted; C uncertain. Choose one.',list('ABC')) if obs else 'B'
                extra=choose(ctx,images,'Assess additional statements beyond the requested answer. A at least one demonstrably false extra detail or explanation; B no clear extra error; C extra details are unverifiable from this image and reference. Choose one.',list('ABC'))
                for b in unique[ans,obs_text]:
                    output[k][b]=dict(support=support,extra=extra)
            tmp=ROOT/'partial.tmp'
            tmp.write_text(json.dumps(output));tmp.replace(ROOT/'private_review.json')
    (ROOT/'complete.json').write_text(json.dumps({'n':len(output)}))

if __name__=='__main__':run()
