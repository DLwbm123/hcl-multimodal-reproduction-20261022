"""Post-evaluation aggregates derived from private paired predictions."""
import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean


def fundus(rows):
    # Each row stores fixed single, fixed fusion, baseline policy, learned policy.
    reductions = [r[0]['error'] - r[1]['error'] for r in rows]
    reviewed = [bool(r[3]['review']) for r in rows]
    fraction = mean(reviewed)
    return {
        'reviewed': sum(reviewed),
        'review_beneficial_all': sum(x > 0 for x in reductions),
        'review_harmful_all': sum(x < 0 for x in reductions),
        'review_beneficial_selected': sum(use and x > 0 for use, x in zip(reviewed, reductions)),
        'review_harmful_selected': sum(use and x < 0 for use, x in zip(reviewed, reductions)),
        'learned_reduction_from_first': mean(x * use for use, x in zip(reviewed, reductions)),
        'expected_reduction_uniform_same_count': fraction * mean(reductions),
        'hindsight_lowest_mae': mean(min(r[0]['error'], r[1]['error']) for r in rows)
    }


def clinical(rows):
    details = {}
    for arm in ('baseline', 'candidate'):
        successes = Counter()
        attempts = Counter()
        empty = Counter()
        for row in rows:
            for call in row[arm]['notes']:
                name = call['name']
                attempts[name] += 1
                if call['ok']:
                    successes[name] += 1
                    if not call['text'].strip():
                        empty[name] += 1
        details[arm] = {'attempts': attempts, 'tool_ok': successes, 'empty_success_text': empty}
    return details


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('study', choices=['fundus', 'clinical'])
    p.add_argument('pairs', type=Path)
    args = p.parse_args()
    rows = list(json.loads(args.pairs.read_text()).values())
    print(json.dumps((fundus if args.study == 'fundus' else clinical)(rows), indent=2))
