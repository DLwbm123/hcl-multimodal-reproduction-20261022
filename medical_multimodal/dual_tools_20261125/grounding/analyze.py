"""Summarize paired decisions from a finished private run without exporting case identifiers."""
import json
import sys
from collections import Counter
from math import comb
from pathlib import Path
from statistics import mean


def exact_pair(gains,losses):
    n=gains+losses
    return min(1,2*sum(comb(n,k) for k in range(min(gains,losses)+1))/2**n)


def panel(rows):
    answer={}
    for arm in ('baseline','candidate'):
        counts=Counter()
        local=Counter()
        routes=Counter()
        multi=0
        for row in rows:
            notes=row[arm]['notes']
            seq=[n['name'] for n in notes if n['ok']]
            counts.update(seq)
            routes[tuple(seq)]+=1
            multi+=int(len(set(seq))>=2)
            regions=[n for n in notes if n['name']=='RegionAttributeDescription']
            local['attempts']+=len(regions)
            local['successes']+=sum(n['ok'] for n in regions)
            local['nonempty']+=sum(n['ok'] and bool(n['text'].strip()) for n in regions)
            local['cases_success']+=int(any(n['ok'] for n in regions))
        answer[arm]=dict(cases_with_multiple_successful_tool_types=multi,success_calls=counts,
                         local=local,successful_routes={' + '.join(seq) or '(none)':n for seq,n in routes.items()})
    gains=sum(r['candidate_score']['correct']>r['base_score']['correct'] for r in rows)
    losses=sum(r['candidate_score']['correct']<r['base_score']['correct'] for r in rows)
    answer.update(gains=gains,losses=losses,exact_p=exact_pair(gains,losses))
    return answer


def retrieval_use(predictions, bank, questions):
    # Match the actual retrieval call, including insertion-order tie breaking.
    from adapt import retrieval
    counts=Counter()
    for key,row in predictions.items():
        notes=row['candidate']['notes']
        for step in range(min(4,len(notes)+1)):
            prefix=notes[:step]
            stage=prefix[-1]['name'] if prefix else 'start'
            visible='Question: '+questions[key]+'\nObserved:\n'+('\n'.join(
                n['name']+': '+n['text'][:450] for n in prefix) or '(none)')
            suggestion=retrieval(visible,stage,bank).rsplit('\nTraining next action: ',1)[1]
            chosen=notes[step]['name'] if step<len(notes) else 'finish'
            counts['decisions']+=1
            counts['agrees_with_retrieval']+=int(chosen==suggestion)
            if suggestion=='RegionAttributeDescription':
                counts['retrieved_local']+=1
                counts['followed_local']+=int(chosen==suggestion)
                counts['stopped_instead_of_local']+=int(chosen=='finish')
    return counts


if __name__=='__main__':
    root=Path(sys.argv[1])
    report={}
    fit=json.loads((root/'private_fit_region.json').read_text())
    for key,rows in [('all',fit),('unclipped',[r for r in fit if not r['clipped']])]:
        scored=[r for r in rows if r['iou'] is not None]
        if scored:
            report['fit_'+key]={'n':len(scored),'iou':mean(r['iou'] for r in scored),
                                'reference_center_iou':mean(r['center'] for r in scored),
                                'candidate_set_best_iou':mean(r['best'] for r in scored)}
    for phase in ('gate','test','search_stress'):
        file=root/('private_'+phase+'.json')
        if file.is_file():
            current=json.loads(file.read_text())
            report[phase]=panel(current.values())
            if phase=='test' and len(sys.argv)>2:
                previous=json.loads((Path(sys.argv[2])/file.name).read_text())
                if current.keys()!=previous.keys():
                    raise ValueError('Paired comparison requires matching case identifiers')
                changes={}
                for arm in ('base_score','candidate_score'):
                    pairs=[(previous[k][arm]['correct'],current[k][arm]['correct']) for k in current]
                    changes[arm]=dict(before=sum(a for a,b in pairs),after=sum(b for a,b in pairs),
                                      gains=sum(b>a for a,b in pairs),losses=sum(a>b for a,b in pairs))
                report['change_from_previous_test']=changes
    if len(sys.argv)>3:
        bank=json.loads((root/'private_learning.json').read_text())['bank']
        tasks=[json.loads(s) for s in Path(sys.argv[3]).read_text().splitlines()]
        questions={str(x['case_id']):x['question'] for x in tasks}
        for phase in ('gate','test','search_stress'):
            file=root/('private_'+phase+'.json')
            if file.is_file():
                report[phase]['retrieval_use']=retrieval_use(json.loads(file.read_text()),bank,questions)
    print(json.dumps(report,indent=2))
