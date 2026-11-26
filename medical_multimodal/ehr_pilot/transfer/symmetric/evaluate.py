"""Freeze the demo gate from validation, then evaluate two policies by V1 replay."""
import collections
import json
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import compare as run


def read(path):
    return json.loads(Path(path).read_text())


def routed(rows, bank):
    return [run.match(run.ROWS[r['id']], bank)['key'] for r in rows]


def freeze():
    plain = run.merged('plain', 'valid')
    demo = run.merged('demo', 'valid')
    keys = routed(plain, read('library.json'))
    assert len(plain) == len(demo) == 77
    counts = collections.defaultdict(collections.Counter)
    for key, base, candidate in zip(keys, plain, demo):
        assert base['id'] == candidate['id']
        counts[key].update(n=1, wins=int(candidate['qualified'] and not base['qualified']),
                           losses=int(base['qualified'] and not candidate['qualified']))
    enabled = sorted(k for k, c in counts.items() if c['wins'] and not c['losses'])
    run.p.save('demo_gate.json', dict(enabled=enabled, counts=counts))


def paired(ref, method):
    return dict(gains=len(method - ref), losses=len(ref - method),
                kept=len(ref & method), net=len(method) - len(ref))


def evaluate():
    plain = run.merged('plain', 'history')
    cases = [r['id'] for r in plain]
    assert len(set(cases)) == len(cases) == 40
    bank = read('library.json')
    keys = routed(plain, bank)
    base = {r['id'] for r in plain if r['qualified']}
    old = {r['id'] for r in read(run.PRIOR/'baseline.json') if r['qualified']}
    assert old == base
    success = {}
    summary = {}
    for name, gate_path in [('demo', 'demo_gate.json'), ('card', 'gate.json')]:
        rows = run.merged(name, 'history')
        assert cases == [r['id'] for r in rows]
        enabled = read(gate_path)['enabled']
        active = [k in enabled for k in keys]
        chosen = [aug if on else b for aug, b, on in zip(rows, plain, active)]
        hits = {r['id'] for r in chosen if r['qualified']}
        success[name] = hits
        summary[name] = dict(enabled=enabled, active=sum(active), correct=len(hits),
                             vs_plain=paired(base, hits), historical_kept=len(old & hits),
                             official=sum(r['official'] for r in chosen),
                             steps=sum(r['steps'] for r in chosen),
                             tool_calls=sum(role=='agent' and text.startswith(('GET ', 'POST '))
                                            for r in chosen for role, text in r['history']))
    demo, card = success['demo'], success['card']
    comparison = paired(demo, card)
    comparison['both_failed'] = len(cases) - len(demo | card)
    assert comparison['gains'] + comparison['losses'] + comparison['kept'] + comparison['both_failed'] == 40
    by_workflow = {}
    for k in range(1, 11):
        group = {i for i in cases if run.d.kind(run.ROWS[i]) == k}
        ds, cs = demo & group, card & group
        by_workflow[k] = dict(n=len(group), demo=len(ds), card=len(cs), **paired(ds, cs))
    grouped = collections.defaultdict(list)
    for i in cases:
        grouped[run.ROWS[i]['eval_MRN']].append(int(i in card) - int(i in demo))
    groups = list(grouped.values())
    rng = random.Random(261127)
    diffs = []
    for _ in range(10000):
        sample = rng.choices(groups, k=len(groups))
        diffs.append(sum(map(sum, sample)) / sum(map(len, sample)))
    diffs.sort()
    assert summary['card']['correct'] == read(os.environ['V1_RESULT'])['arms']['selected']['correct']
    run.p.save('comparison.json', dict(n=len(cases), patients=len(groups), arms=summary,
               card_vs_demo=comparison, by_workflow=by_workflow,
               rate_diff_patient_bootstrap_95=[diffs[249], diffs[9749]]))


if __name__ == '__main__':
    {'freeze': freeze, 'evaluate': evaluate}[sys.argv[1]]()
