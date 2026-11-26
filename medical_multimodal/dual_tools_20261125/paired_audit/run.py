"""Replay an incumbent stop and its cached check with matched answer generation."""
import json
import os
import random
import time
from pathlib import Path

from med import main
from grounding import locate
from observation import LOCAL, visual_tools
from shared import ROOT, calls, save


def load(directory, filename):
    return json.loads((Path(directory) / filename).read_text())


def experiment(cases, refs, parts, context, original, policy, available, visible,
               answer_from, correct, choose, gen):
    ids = sorted(parts[2])
    anchors = load(os.environ['QUALITY_RUN'], 'private_test.json')
    decisions = load(os.environ['FEATURE_RUN'], 'private_test.json')
    assert set(ids) == set(anchors) == set(decisions) and len(ids) == 45
    assert set(load(ROOT, 'private_rubrics.json')) == set(ids)
    call = visual_tools(original, visible, choose, gen)
    resume = os.environ.get('RESUME_ROOT')
    rows = load(resume, 'private_partial.json') if resume else {}
    assert set(rows) == set(ids[:len(rows)])
    rng = random.Random(265144)
    items = []
    for i, k in enumerate(ids, 1):
        if k in rows:
            unique = list(dict.fromkeys(rows[k]['answers'].values()))
            rng.shuffle(unique)
            items.append(dict(item=i, answers=unique))
            continue
        q, img, path = context(k)
        anchor, old = anchors[k], decisions[k]
        notes = anchor['observations']
        assert len(notes) == anchor['steps'] == old['calls'] <= 4
        events = anchor['events'][:len(notes)]
        assert [e['tool'] for e in events] == [n['name'] for n in notes]
        assert anchor['answer'] == old['answer'] and not old['asked']
        trusted = [n for n, e in zip(notes, events) if e['accepted']]
        s = old['answer']
        r = v = s
        cue = old.get('cue')
        local = None
        times = {}
        if len(notes) < 4:
            assert cue and cue.strip().rstrip('.').upper() != 'NONE'
            t = time.monotonic()
            r = answer_from(q, trusted)
            times['R'] = time.monotonic() - t
            t = time.monotonic()
            local = call(q + '\nInspect this specific visual distinction: ' + cue,
                         img, path, notes, LOCAL)
            v = answer_from(q, trusted + [local]) if local['ok'] else r
            times['V'] = time.monotonic() - t
        answers = dict(S=s, R=r, V=v)
        unique = list(dict.fromkeys(answers.values()))
        rng.shuffle(unique)
        items.append(dict(item=i, answers=unique))
        rows[k] = dict(item=i, answers=answers, local=local, cue=cue,
                       trusted=trusted, steps=len(notes), seconds=times)
        save('private_partial', rows)
        if i % 5 == 0:
            print('finished', i, flush=True)
    count = sum(r['local'] is not None for r in rows.values())
    base = sum(r['steps'] for r in rows.values())
    assert (count, base) == (44, 53)
    rng.shuffle(items)
    save('private_blind', items)
    save('private_branches', rows)
    save('runtime', dict(cases=len(rows), old_calls=base, extra_calls=count,
                         valid=sum(bool(r['local'] and r['local']['ok']) for r in rows.values()),
                         truncated=sum(bool(r['local'] and len(r['local']['text']) > 450) for r in rows.values()),
                         model_calls=dict(calls),
                         branch_seconds={a:sum(r['seconds'].get(a, 0) for r in rows.values()) for a in ('R','V')}))


if __name__ == '__main__':
    main(locator=locate, experiment=experiment)
