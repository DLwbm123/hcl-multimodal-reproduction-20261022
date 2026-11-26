"""Post-hoc patient-grouped replay of two frozen development trajectories."""
import collections
import json
import random
import re
import sys
from pathlib import Path


def words(text):
    return set(re.findall(r'[a-z]+', re.sub(r'\bS\d+\b', '', text).lower()))


def review(folder, cases):
    rows = {name: {r['id']: r for r in json.loads((folder / (name + '.json')).read_text())}
            for name in ('plain', 'guide')}
    assert rows['plain'].keys() == rows['guide'].keys() and len(rows['plain']) == 160
    cases = [c for c in cases if c['id'] in rows['plain']]
    patients = sorted({c['eval_MRN'] for c in cases})
    random.Random(261126).shuffle(patients)
    folds = {mrn: i % 5 for i, mrn in enumerate(patients)}
    counts = collections.Counter()
    by_fold = []
    for fold in range(5):
        training = [c for c in cases if folds[c['eval_MRN']] != fold]
        held = [c for c in cases if folds[c['eval_MRN']] == fold]
        changes = collections.defaultdict(list)
        examples = []
        for c in training:
            key = c['id'].split('_')[0]
            changes[key].append(int(rows['guide'][c['id']]['qualified']) -
                                int(rows['plain'][c['id']]['qualified']))
            examples.append((words(c['instruction']), key))
        allowed = {key for key, ds in changes.items() if sum(ds) > 0 and min(ds) >= 0}
        part = collections.Counter()
        for c in held:
            q = words(c['instruction'])
            _, route = max(examples, key=lambda e:len(e[0] & q) / max(1,len(e[0] | q)))
            # The cached guide was run with family routing. A mismatched route
            # has no corresponding observed outcome and must not be imputed.
            if route != c['id'].split('_')[0]:
                raise ValueError('Cached guide cannot evaluate predicted route')
            active = route in allowed
            b = rows['plain'][c['id']]['qualified']
            proposed = rows['guide' if active else 'plain'][c['id']]['qualified']
            part.update(n=1, used=int(active), base=int(b), correct=int(proposed),
                        gains=int(proposed and not b), losses=int(b and not proposed))
        by_fold.append(dict(part))
        counts.update(part)
    return dict(total=counts, patient_count=len(patients), folds=by_fold)


if __name__ == '__main__':
    folder = Path(sys.argv[1])
    print(json.dumps(review(folder,json.loads(Path(sys.argv[2]).read_text())),indent=2))
