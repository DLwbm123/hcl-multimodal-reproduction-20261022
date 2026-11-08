"""Atomic deployment pointer; transient candidate memory is never recovery state."""
from copy import deepcopy
import json
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + '\n')
    temp.replace(path)


class StateStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, name, state):
        path = self.root / (name + '.json')
        if path.exists():
            raise FileExistsError('State names are immutable: ' + name)
        write_json(path, state)
        return path

    def deploy(self, name, state, parent, decision):
        # Write a complete snapshot before the atomic pointer becomes visible.
        self.save(name, state)
        write_json(self.root / 'DEPLOYED.json',
                   {'name': name, 'parent': parent, 'decision': decision})
        restored = self.load()
        if restored != state:
            raise RuntimeError('Deployed snapshot restore mismatch')
        return restored

    def load(self):
        pointer = json.loads((self.root / 'DEPLOYED.json').read_text())
        return json.loads((self.root / (pointer['name'] + '.json')).read_text())


def gate(reference, candidate, gold, expected_ids, anchor_reference=(),
         anchor_candidate=(), anchor_gold=None, anchor_ids=()):
    """Success-to-failure losses cannot be offset by recovered anchors."""
    reasons = []
    def check(rows, ids, label):
        seen = [r['id'] for r in rows]
        if len(seen) != len(set(seen)) or set(seen) != set(ids):
            reasons.append(label + '_membership')
        if any(r.get('status') != 'completed' or r.get('predicted') not in list('ABCDEF')
               for r in rows):
            reasons.append(label + '_invalid')
        return {r['id']: r.get('predicted') for r in rows}
    before = check(reference, expected_ids, 'reference')
    after = check(candidate, expected_ids, 'candidate')
    correct_before = sum(before.get(k) == gold[k] for k in expected_ids)
    correct_after = sum(after.get(k) == gold[k] for k in expected_ids)
    if correct_after - correct_before < 1:
        reasons.append('no_positive_current_gain')
    protected = losses = recovered = 0
    if anchor_ids:
        old = check(anchor_reference, anchor_ids, 'anchor_reference')
        new = check(anchor_candidate, anchor_ids, 'anchor_candidate')
        for k in anchor_ids:
            old_ok, new_ok = old.get(k) == anchor_gold[k], new.get(k) == anchor_gold[k]
            protected += int(old_ok)
            losses += int(old_ok and not new_ok)
            recovered += int(not old_ok and new_ok)
        if losses:
            reasons.append('historical_correct_to_wrong_loss')
    return {'accept': not reasons, 'reasons': reasons,
            'current_denominator': len(expected_ids), 'before_correct': correct_before,
            'after_correct': correct_after, 'current_gain': correct_after - correct_before,
            'anchor_denominator': len(anchor_ids), 'anchor_protected': protected,
            'anchor_losses': losses, 'anchor_recovered': recovered}


def paired(before, after, ids, gold):
    a = {r['id']: r.get('predicted') for r in before}
    b = {r['id']: r.get('predicted') for r in after}
    return {'denominator': len(ids),
            'gain': sum(a.get(k) != gold[k] and b.get(k) == gold[k] for k in ids),
            'loss': sum(a.get(k) == gold[k] and b.get(k) != gold[k] for k in ids)}


def state_diff(before, after):
    changed = [k for k in before if before[k] != after.get(k)]
    return {'changed_top_level_fields': changed,
            'memory_counts_before': {k: len(v) for k, v in before['memory'].items()},
            'memory_counts_after': {k: len(v) for k, v in after['memory'].items()},
            'selector_changed': before['memory_selector'] != after['memory_selector']}
