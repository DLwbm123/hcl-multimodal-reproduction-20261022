import json
import os
import random
from collections import defaultdict
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn


def signature(path):
    with Image.open(path) as im:
        x = np.asarray(ImageOps.grayscale(im).resize((32, 32), Image.Resampling.LANCZOS), dtype=float)
    block = dctn(x, norm='ortho')[:8, :8].ravel()[1:]
    return int(''.join('1' if b else '0' for b in block > np.median(block)), 2)


def freeze():
    root=Path(os.environ['EXTERNAL_ROOT'])
    old=Path(os.environ['DATA_ROOT'])
    dest=Path(os.environ['PANEL_ROOT'])
    rows=json.loads((root/'VQA_RAD Dataset Public.json').read_text())
    parent={}
    def group(x):
        parent.setdefault(x,x)
        while x!=parent[x]:
            parent[x]=parent[parent[x]]
            x=parent[x]
        return x
    entries=[]
    for r in rows:
        name=r['image_name'].strip()
        case=parse_qs(urlparse(r['image_case_url'].strip()).query).get('id',[''])[0].lower()
        if not name or not case:
            continue
        parent[group(name)]=group(case)
        if r['phrase_type'] in ('freeform','test_freeform') and r['question'].strip() and str(r['answer']).strip().lower() not in ('','null','none provided'):
            entries.append((name,r))
    grouped=defaultdict(list)
    for name,r in entries:
        grouped[group(name)].append((name,r))
    oldfiles={p for line in (old/'tasks.jsonl').read_text().splitlines() for p in json.loads(line)['images']}
    assert len(oldfiles)==107
    history=[signature(old/f) for f in sorted(oldfiles)]
    rng=random.Random(2026112601)
    groups=sorted(grouped)
    rng.shuffle(groups)
    inputs=[]; labels={}; orig=[]; rejected=0
    for g in groups:
        candidates=sorted(grouped[g],key=lambda pair:str(pair[1]['qid']))
        name,r=rng.choice(candidates)
        path=root/'images'/name
        h=signature(path)
        if any((h^v).bit_count()<=10 for v in history):
            rejected+=1
            continue
        history.append(h)
        k=str(len(inputs)+1)
        inputs.append(dict(id=k,path=str(path),q=r['question'].strip()))
        labels[k]=str(r['answer']).strip()
        orig.append(dict(id=k,image=name,case=g,qid=r['qid'],split=r['phrase_type']))
        if len(inputs)==200:break
    assert len(inputs)==len(set(r['path'] for r in inputs))==200
    for file,data in [('inputs',inputs),('labels',labels),('sources',orig),('panel',dict(n=200,old_images=len(oldfiles),groups=len(groups),skipped_similar=rejected))]:
        with (dest/(file+'.json')).open('x') as f: json.dump(data,f)
    print((dest/'panel.json').read_text())


if __name__=='__main__':freeze()
