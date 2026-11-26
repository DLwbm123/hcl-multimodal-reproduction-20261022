"""Unblind frozen adjudications and bound unknowns using identical answers jointly."""
import itertools
import json
import sys
from collections import Counter
from pathlib import Path


def summarize(root):
    def read(name):
        return json.loads((root/(name + '.json')).read_text())
    blind = {r['item']: r['answers'] for r in read('private_blind')}
    grades = read('private_grades')
    branches = read('private_branches')
    totals = {b: Counter() for b in 'SRV'}
    pairs = {p: Counter() for p in ('VS','VR','RS')}
    candidates = []
    table = []
    for row in branches.values():
        i = row['item']
        labels = grades[str(i)]
        assert len(labels) == len(blind[i]) and set(labels) <= set('01U')
        scores = dict(zip(blind[i],labels))
        answers = row['answers']
        outcome = {b:scores[a] for b,a in answers.items()}
        for b,s in outcome.items():
            totals[b][s] += 1
        if outcome == dict(S='0', R='0', V='1'):
            candidates.append(i)
        for p, stats in pairs.items():
            a,b = (answers[k] for k in p)
            unknown = [x for x in set((a,b)) if scores[x]=='U']
            changes = []
            for assignment in itertools.product('01',repeat=len(unknown)):
                completed = {**scores, **dict(zip(unknown,assignment))}
                changes.append(int(completed[a])-int(completed[b]))
            stats['low'] += min(changes)
            stats['high'] += max(changes)
            stats['gain'] += min(changes)==1
            stats['loss'] += max(changes)==-1
            stats['unresolved'] += min(changes)!=max(changes)
        table.append(dict(item=i, **outcome, eligible=row['local'] is not None))
    assert len(table)==len(blind)==len(grades)==45
    return dict(scores=totals, pairs=pairs, unique_visual_candidates=candidates), table


if __name__ == '__main__':
    root=Path(sys.argv[1])
    result, table=summarize(root)
    for name,data in [('result',result),('paired_table',table)]:
        (root/(name+'.json')).write_text(json.dumps(data,indent=2)+'\n')
