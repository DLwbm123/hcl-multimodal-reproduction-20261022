import json
import re
from pathlib import Path
from prep import read_rows, select


def build(raw, prior_dirs):
    pool, _, _ = select(raw)
    used = set()
    for split in ("development","heldout"):
        used.update(r["case"] for r in read_rows(raw/"pilot"/split/"inputs.jsonl"))
    for folder in prior_dirs:
        for stage in ("A","B"):
            for role in ("fit","gate","confirm"):
                used.update(p.name for p in (folder/(stage+"_"+role)/"images").iterdir() if p.is_dir())
    if len(used)!=480:
        raise ValueError(f"Expected 480 distinct historical cases, found {len(used)}")
    picked = []
    for has_ct in (True,False):
        eligible = (r for r in pool if r[0] not in used and len(r[-1])<=27 and
                    any(v["modality"]=="CT" for v in r[-1])==has_ct)
        picked.extend((next(eligible),has_ct) for _ in range(32))
    captions = {}
    paths = {v["path"] for r,_ in picked for v in r[-1][:3]}
    for r in read_rows(raw/"train.jsonl"):
        for k,path in r.items():
            if re.fullmatch(r"image_\d{2}_path",k) and path in paths:
                captions[path]=r.get(k[:-4]+"caption","") or ""
    if set(captions)!=paths:
        raise ValueError("Caption slot mapping incomplete")
    cases,gold={},{}
    for (k,_,history,options,label,views),ct in picked:
        cases[k]=dict(stage="CT" if ct else "nonCT",history=history,options=options,
                      images=[dict(path=v["path"],caption=captions[v["path"]]) for v in views[:3]])
        gold[k]=label
    return cases,gold
