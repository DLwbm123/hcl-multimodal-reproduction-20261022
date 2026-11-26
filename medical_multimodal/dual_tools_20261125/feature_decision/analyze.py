"""Report paired results without exposing images, observations, answers or identifiers."""
import json
import sys
from collections import Counter
from math import comb
from pathlib import Path


def load(root, name):
    return json.loads((root / name).read_text())


def summarize(rows, prior, keys):
    counts = Counter()
    for key in keys:
        r = rows[key]
        old, new = prior[key]['revised'], r['correct']
        counts['asked'] += r['asked']
        counts['adopted'] += r['adopted']
        counts['verified_quote'] += bool(r.get('quote'))
        counts['changed_answer'] += r['answer'] != prior[key]['answer']
        counts['gain'] += new > old
        counts['loss'] += new < old
        counts['retained'] += new == old == 1
        counts['failed_request'] += r['asked'] and not r['local']['ok']
    changed = counts['gain'] + counts['loss']
    p = min(1, 2 * sum(comb(changed, k) for k in range(min(counts['gain'], counts['loss'])+1)) / 2**changed)
    return dict(n=len(keys), **counts, exact_p=p)


if __name__ == '__main__':
    root, previous = map(Path, sys.argv[1:])
    report = {}
    training = load(root, 'private_fit.json')
    report['training'] = dict(quoted_and_corrected=sum(bool(r['quote']) and r['post']>r['pre'] for r in training),
                              lost_on_deletion=sum(r['post']>r['pre'] and r['without']==0 for r in training),
                              correct_with_phrase_only=sum(r['post']>r['pre'] and r['only']==1 for r in training))
    for phase in ('gate','test'):
        rows = load(root,'private_'+phase+'.json')
        prior = load(previous,'private_'+phase+'.json')
        assert set(rows)==set(prior)
        report[phase] = {group:summarize(rows,prior,[k for k in rows if predicate(k)]) for group,predicate in [
            ('all', lambda k: True), ('previous_correct', lambda k: prior[k]['revised']==1),
            ('disputed', lambda k: prior[k]['disputed'])]}
    print(json.dumps(report,indent=2))
