"""Challenge an early stop once when the frozen memory explicitly requests unseen local evidence."""
import json
import os
from collections import Counter
from pathlib import Path
from adapt import metrics, retrieval
from clinical import Backend, DATA, actions, data, evaluate
from grounding import locate
from PIL import Image
from shared import models, save


def state(q,notes):
    return 'Question: '+q+'\nObserved:\n'+('\n'.join(n['name']+': '+n['text'][:450] for n in notes) or '(none)')


def challenge(q,notes,bank):
    if not notes or len(notes)>=4 or any(n['name']=='RegionAttributeDescription' for n in notes):
        return False
    hint=retrieval(state(q,notes),notes[-1]['name'],bank)
    return hint.endswith('\nTraining next action: RegionAttributeDescription')


def run():
    root=Path(os.environ['PAIRED_RUN'])
    bank=json.loads((root/'private_learning.json').read_text())['bank']
    cases,refs=data()
    choose,gen=models()
    backend=Backend()
    result={}
    try:
        for phase in ('gate','search_stress'):
            rows=json.loads((root/('private_'+phase+'.json')).read_text())
            scores=[];wins=Counter();reviews=0
            for k,row in rows.items():
                q=cases[k]['question']
                notes=row['candidate']['notes']
                s=dict(row['candidate_score'])
                if challenge(q,notes,bank):
                    reviews+=1
                    path=DATA/cases[k]['images'][0]
                    with Image.open(path) as source:
                        box=locate(choose,state(q,notes),source.convert('RGB'),set())
                    out=backend.call(dict(name='RegionAttributeDescription',path=str(path),box=box,
                                          attribute='Visible features relevant to: '+q))
                    notes=notes+[dict(name='RegionAttributeDescription',ok=out['ok'],text=out['text'])]
                    answer=gen(state(q,notes)+'\nGive the concise clinical answer supported by this evidence.',96)
                    label=choose('Reference acceptance criteria: '+json.dumps(refs[k]['gt_answer'])+'\nAnswer: '+answer,{},
                                 'A = answer satisfies the reference clinically including key details, B = incorrect or incomplete. Return A or B.',list('AB'))
                    seq=[n['name'] for n in notes if n['ok']]
                    expected=actions(refs[k])
                    s=dict(correct=int(label=='A'),f1=evaluate(seq,expected),exact=int(seq==expected),
                           steps=len(notes),invalid=sum(not n['ok'] for n in notes))
                for old in ('base_score','candidate_score'):
                    wins[old+'_gain']+=int(s['correct']>row[old]['correct'])
                    wins[old+'_loss']+=int(s['correct']<row[old]['correct'])
                scores.append(s)
            summary=metrics(scores)
            base=metrics([r['base_score'] for r in rows.values()])
            cand=metrics([r['candidate_score'] for r in rows.values()])
            result[phase]=dict(n=len(rows),reviews=reviews,baseline=base,original_candidate=cand,
                               reviewed_candidate=summary,paired=wins)
            if phase=='gate':
                result['accepted']=summary['f1']>base['f1'] and summary['correct']>=base['correct']
            save('stop_review_result',result)
            print(phase,len(rows),reviews,flush=True)
            if not result['accepted']:
                break
    finally:
        backend.close()


if __name__=='__main__':
    run()
