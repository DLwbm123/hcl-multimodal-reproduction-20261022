"""Allocate the fresh VitalDB panel without exposing member IDs in the receipt."""
import json
from pathlib import Path
import random

from environment import needs_more
from panel import FORMS, view
from prepare import eligible_cases

SEED = 26112892
SPLITS = ('train', 'dev', 'test')


def read(path):
    return json.loads(path.read_text())


def main():
    paths = (Path('p0'), Path('p1'), Path('p2'))
    subjects = [{int(r['subject_id']) for s in SPLITS for r in read(p/(s+'.json'))} for p in paths]
    assert [len(s) for s in subjects] == [224, 224, 160]
    excluded = set.union(*subjects)
    assert len(excluded) == 608
    # The original reconstruction internally excludes the archived 120+44 cohort,
    # checking the 44 historical receipt series against the reconstructed rows.
    eligible = eligible_cases(Path('data'), read(Path('legacy.json')), excluded)
    rng = random.Random(SEED)
    groups = {True: [], False: []}
    for case, (subject, rows) in sorted(eligible.items()):
        available = sum(age <= 24 for age, _ in rows)
        if available < 2 or len({age for age, _ in rows}) != len(rows):
            continue
        sufficient = available if rows[available-1][0]==24 or available==len(rows) else available+1
        if sufficient > 16:
            continue
        insufficient = rng.randint(1, min(4, available-1))
        groups[sufficient < len(rows)].append(dict(subject_id=subject, case_id=case, rows=rows,
                                                  insufficient=insufficient, sufficient=sufficient))
    for pool in groups.values():
        rng.shuffle(pool)
        assert len(pool) >= 32
    test = [pool[i] for i in range(32) for pool in groups.values()]
    known = {q for split in FORMS.values() for q in split}
    extra = read(Path('forms.json'))
    assert len(extra) == 2 and len(set(extra))==2 and not known.intersection(extra)
    assert len({p['subject_id'] for p in test})==64 and not ({p['subject_id'] for p in test}&excluded)
    assert all(needs_more(view(p,'insufficient',FORMS['test'][0])) and
               not needs_more(view(p,'sufficient',FORMS['test'][0])) for p in test)
    out=Path('new_private.json')
    with out.open('x') as f:
        json.dump(test,f,allow_nan=False)
    out.chmod(0o400)
    result=dict(seed=SEED, patients=64, archived_subject_id_union=len(excluded),
                additionally_reconstructed=164, total_excluded=len(excluded)+164,
                required_inputs=384, sufficient_with_next_inputs=192, sufficient_last_inputs=192,
                old_forms=4, new_forms=2, verified_disjoint=True)
    Path('panel_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
