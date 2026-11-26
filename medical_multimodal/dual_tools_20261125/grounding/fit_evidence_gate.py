"""Fit-only leave-one-case-out probe of whether similar first-region states predict answer gains."""
import json
import sys
from collections import Counter
from pathlib import Path
from previous_learning import words


def probe(fit, bank, observations):
    groups=[]
    for row in bank:
        if row['stage']=='start':
            groups.append([])
        groups[-1].append(row)
    assert len(groups)==len(fit)
    states={k:words(s['visible']) for k,group in zip(fit,groups)
            for s in [next((r for r in group if r['next']=='RegionAttributeDescription'),None)]
            if s is not None}
    scores={r['case']:r for r in observations}
    assert states.keys()==scores.keys() and len(states)>5
    count=Counter()
    for k,query in states.items():
        ranked=sorted([(len(query & x)/max(1,len(query | x)),v)
                       for v,x in states.items() if v!=k], reverse=True)
        threshold=ranked[4][0]
        peers=[v for similarity,v in ranked if similarity>=threshold]
        use=sum(scores[v]['after']-scores[v]['before'] for v in peers)>0
        row=scores[k]
        count['n']+=1
        count['used']+=int(use)
        count['correct_without']+=row['before']
        count['correct_always']+=row['after']
        count['correct_selected']+=row['after'] if use else row['before']
        count['gains_vs_without']+=int(use and row['after']>row['before'])
        count['losses_vs_without']+=int(use and row['after']<row['before'])
    return count


if __name__=='__main__':
    from clinical import split
    root=Path(sys.argv[1])
    print(json.dumps(probe(split()[0],json.loads((root/'private_learning.json').read_text())['bank'],
                           json.loads((root/'private_fit_region.json').read_text())),indent=2))
