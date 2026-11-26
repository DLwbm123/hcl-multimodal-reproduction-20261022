"""Controlled replay on fitting cases only. Study reference boxes and tool prompts separately."""
import json
import os
import time
from pathlib import Path
from PIL import Image
from clinical import Backend, DATA, data, split
from grounding import reference_box
from shared import models, save


def run():
    parent=Path(os.environ['PAIRED_RUN'])
    cases,refs=data()
    fit,_,_=split()
    choose,gen=models()
    backend=Backend()
    try:
        while not (parent/'private_learning.json').is_file():
            if (parent/'SUPERVISOR_EXIT.json').exists():
                raise RuntimeError('parent exited before publishing fit memory')
            time.sleep(3)
        bank=json.loads((parent/'private_learning.json').read_text())['bank']
        groups=[]
        for row in bank:
            if row['stage']=='start':
                groups.append([])
            groups[-1].append(row)
        assert len(groups)==len(fit)
        predicted={r['case']:r for r in json.loads((parent/'private_fit_region.json').read_text())}
        rows=[]
        for key,group in zip(fit,groups):
            if key not in predicted:
                continue
            case=cases[key]
            prefix=next(x['visible'] for x in group if x['next']=='RegionAttributeDescription')
            assert prefix.startswith('Question: '+case['question']+'\nObserved:\n')
            args=next(t['function']['arguments'] for m in refs[key]['dialogs'] for t in m.get('tool_calls',[])
                      if t['function']['name']=='RegionAttributeDescription')
            args=json.loads(args) if isinstance(args,str) else args
            path=DATA/case['images'][0]
            with Image.open(path) as im:
                box,_=reference_box(args,im.size)
            if box is None or not args.get('attribute'):
                continue
            row={'before':predicted[key]['before'],'predicted_box':predicted[key]['after']}
            for mode,attribute in [('reference_box','Visible features relevant to: '+case['question']),
                                   ('reference_box_and_attribute',args['attribute'])]:
                out=backend.call(dict(name='RegionAttributeDescription',path=str(path),box=box,attribute=attribute))
                assert out['ok']
                state=prefix.removesuffix('(none)') if prefix.endswith('Observed:\n(none)') else prefix+'\n'
                answer=gen(state+'RegionAttributeDescription: '+out['text'][:450]+'\nGive the concise clinical answer supported by this evidence.',96)
                label=choose('Reference acceptance criteria: '+json.dumps(refs[key]['gt_answer'])+'\nAnswer: '+answer,{},
                             'A = answer satisfies the reference clinically including key details, B = incorrect or incomplete. Return A or B.',list('AB'))
                row[mode]=int(label=='A')
            rows.append(row)
            if len(rows)%5==0:
                print('fit pairs',len(rows),flush=True)
        if not rows:
            raise ValueError('no eligible reference crops')
        totals={k:sum(r[k] for r in rows) for k in rows[0]}
        paired={}
        for a,b in [('before','predicted_box'),('predicted_box','reference_box'),
                    ('reference_box','reference_box_and_attribute')]:
            paired[b]={'gain':sum(r[b]>r[a] for r in rows),'loss':sum(r[b]<r[a] for r in rows)}
        save('fit_controls',dict(n=len(rows),correct=totals,paired=paired))
    finally:
        backend.close()


if __name__=='__main__':
    run()
