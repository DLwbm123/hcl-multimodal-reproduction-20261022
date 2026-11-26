"""Public paired aggregates; do not emit original questions, answers or case IDs."""
import os,sys,json
from pathlib import Path
from math import comb


def load(root,name):return json.loads((root/name).read_text())


def paired(a,b):
    gain=sum(x<y for x,y in zip(a,b));loss=sum(x>y for x,y in zip(a,b));n=gain+loss
    return dict(before=sum(a),after=sum(b),gain=gain,loss=loss,
                exact_p=min(1,2*sum(comb(n,k) for k in range(min(gain,loss)+1))/2**n))


def summarize(old,new,ids):
    a=[old[k] for k in ids];b=[new[k] for k in ids]
    new_scores=[r['revised'] for r in b]
    return dict(n=len(ids),vs_previous=paired([r['revised'] for r in a],new_scores),
                vs_base=paired([r['baseline'] for r in a],new_scores),
                old_calls=sum(r['steps'] for r in a),new_calls=sum(r['steps'] for r in b),
                previous_correct_retained=sum(x['revised'] and y['revised'] for x,y in zip(a,b)))


if __name__=='__main__':
    current,previous,fixed=map(Path,sys.argv[1:])
    report={}
    for phase in ('gate','test'):
        old=load(previous,'private_'+phase+'.json');new=load(current,'private_'+phase+'.json')
        assert set(old)==set(new) and all(1<=r['steps']<=4 and len(r['observations'])==r['steps'] for r in new.values())
        groups={'all':list(old)}
        if phase=='test':
            groups.update(disputed=[k for k in old if old[k]['disputed']],
                          continued_errors=[k for k in old if old[k]['disputed'] and old[k]['steps']>1 and not old[k]['revised']],
                          other=[k for k in old if not old[k]['disputed']])
        report[phase]={g:summarize(old,new,ids) for g,ids in groups.items()}
        if phase=='test':
            replay=load(fixed,'private_fixed.json');ids=groups['continued_errors']
            assert len(ids)==19 and set(ids)==set(replay)
            report['fixed_vs_closed']=paired([replay[k]['correct'] for k in ids],[new[k]['revised'] for k in ids])
    print(json.dumps(report,indent=2))
