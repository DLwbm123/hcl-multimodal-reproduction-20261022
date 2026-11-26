"""Fixed follow-up on every eligible stop with no accepted observation."""
import json
import os
import random

from grounding import locate
from med import main
from observation import GLOBAL, visual_tools
from shared import ROOT, calls, save


def experiment(cases, refs, parts, context, original, policy, available, visible,
               answer_from, correct, choose, gen):
    from pathlib import Path
    anchors=json.loads((Path(os.environ['QUALITY_RUN'])/'private_test.json').read_text())
    call=visual_tools(original,visible,choose,gen)
    records={}
    blinded=[]
    rng=random.Random(765114)
    for i,k in enumerate(sorted(parts[2]),1):
        anchor=anchors[k]
        notes=anchor['observations']
        if len(notes)>=4 or any(e['accepted'] for e in anchor['events'][:len(notes)]):
            continue
        q,img,path=context(k)
        assert len(notes)==1 and notes[0]['ok']
        old=answer_from(q,notes)
        cached=notes[0]['name']==GLOBAL
        whole=notes[0] if cached else call(q,img,path,notes,GLOBAL)
        whole_answer=old if cached else answer_from(q,[whole])
        records[k]=dict(item=i,retained=old,global_answer=whole_answer,cached=cached)
        values=list(dict.fromkeys([old,whole_answer]))
        rng.shuffle(values)
        blinded.append(dict(item=i,answers=values))
    assert len(records)==7
    save('private_control',records)
    save('private_blind',blinded)
    save('runtime',dict(n=len(records),fresh_global=sum(not x['cached'] for x in records.values()),model_calls=dict(calls)))


if __name__=='__main__':
    main(locator=locate,experiment=experiment)
