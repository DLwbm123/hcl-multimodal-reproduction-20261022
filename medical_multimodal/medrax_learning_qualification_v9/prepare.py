"""Outcome-blind allocation, with a protected metadata-only confirmation reserve."""
import collections
import json
import os
from pathlib import Path
import random
import time
from PIL import Image

ROOT = Path(os.environ['R981_ROOT'])
BASE = Path(os.environ['R996_BASE'])


def main():
    if (ROOT / 'private/members.json').exists():
        raise RuntimeError('Allocation already frozen; no resampling')
    plan = json.loads((ROOT / 'code/protocol.json').read_text())
    metadata = BASE / 'data/chestagentbench/metadata.jsonl'
    fields = ['full_question_id', 'case_id', 'images', 'image_source_urls', 'categories']
    rows = [{k: r[k] for k in fields} for r in map(json.loads, metadata.read_text().splitlines())]
    image_root = BASE / 'data/chestagentbench/images'
    def identities(row):
        return {str((image_root / x).resolve()) for x in row['images']} | set(row['image_source_urls'])
    excluded = set()
    sources = [('medrax_v1', 'native_results.private.json'), ('medrax_tool_consult_v2', 'results.private.json'),
               ('medrax_tool_value_v3', 'results.private.json'), ('react_vision_v1', 'results.private.json'),
               ('medrax_subscription_v4', 'private/panel.json')]
    sources += [(n, 'private/members.json') for n in ['medrax_hcl_v5', 'medrax_memory_interface_v6', 'medrax_corrective_memory_v7', 'medrax_error_abstraction_v8']]
    for name, rel in sources:
        excluded.update(str(r['case_id']) for r in json.loads((BASE.parent / name / rel).read_text()))
    assert len(excluded) == 300
    forbidden = set().union(*(identities(r) for r in rows if str(r['case_id']) in excluded))
    eligible = [r for r in rows if str(r['case_id']) not in excluded and not identities(r) & forbidden]
    b_cases = {str(r['case_id']) for r in eligible if 2 <= len(r['images']) <= 4 and 'comparison' in r['categories']}
    groups = {s: collections.defaultdict(list) for s in ['A', 'B']}
    for row in eligible:
        case = str(row['case_id']); n = len(row['images'])
        stage = 'B' if case in b_cases and 2 <= n <= 4 and 'comparison' in row['categories'] else 'A' if case not in b_cases and n == 1 and 'diagnosis' in row['categories'] else None
        if stage: groups[stage][case].append(row)
    rng = random.Random(plan['selection']['seed'])
    used_images = set(forbidden); used_cases = set(excluded); members = []
    for role, count in [('confirmation_reserved', 48), ('train', 24)]:
        for stage in ['A', 'B']:
            keys = sorted(groups[stage]); rng.shuffle(keys); selected = []
            for case in keys:
                if case in used_cases: continue
                options = sorted(groups[stage][case], key=lambda r: r['full_question_id']); rng.shuffle(options)
                for row in options:
                    if identities(row) & used_images: continue
                    selected.append({**row, 'stage': stage, 'role': role})
                    used_cases.add(case); used_images.update(identities(row)); break
                if len(selected) == count: break
            assert len(selected) == count, 'Insufficient disjoint cases; no label-informed fallback'
            members.extend(selected)
    old_inputs = []; old_supervision = {}
    for name in ['medrax_hcl_v5', 'medrax_corrective_memory_v7']:
        prior = BASE.parent / name
        old_inputs.extend(r for r in json.loads((prior / 'private/inputs.json').read_text()) if r['role'] == 'train')
        old_supervision.update(json.loads((prior / 'private/supervision.json').read_text()))
    repeated = []
    for stage in ['A', 'B']:
        options = sorted([r for r in old_inputs if r['stage'] == stage], key=lambda r: (r['case_id'], r['id']))
        rng.shuffle(options); repeated.extend([{**r, 'role': 'repeat_development'} for r in options[:12]])
    assert len(repeated) == len({r['case_id'] for r in repeated}) == 24
    (ROOT / 'private/members.json').write_text(json.dumps(members, indent=2))
    (ROOT / 'private/repeat_members.json').write_text(json.dumps([{k: r[k] for k in ['id', 'case_id', 'stage', 'role']} for r in repeated]))
    (ROOT / 'FROZEN_MEMBERS.json').write_text(json.dumps({'time': time.time(), 'selection_used_labels': False, 'excluded_cases': 300, 'new_train_cases': 48, 'repeat_development_cases': 24, 'confirmation_reserved_cases': 96}))
    train_members = [r for r in members if r['role'] == 'train']
    selected_ids = {r['full_question_id'] for r in train_members}
    # Only selected training records retain labels; reserved-case labels are never accessed.
    lookup = {r['full_question_id']: r for r in map(json.loads, metadata.read_text().splitlines()) if r['full_question_id'] in selected_ids}
    train = []; gold = {r['id']: old_supervision[r['id']] for r in repeated}
    for member in train_members:
        row = lookup[member['full_question_id']]; key = row['full_question_id']; answer = str(row['answer']).strip()
        assert answer in list('ABCDEF')
        paths = [str((image_root / x).resolve()) for x in row['images']]
        train.append({'id': key, 'case_id': str(row['case_id']), 'stage': member['stage'], 'role': 'train', 'question': row['question'], 'images': paths})
        gold[key] = answer
    interleave = lambda items: [x for pair in zip([r for r in items if r['stage'] == 'A'], [r for r in items if r['stage'] == 'B']) for x in pair]
    inputs = interleave(repeated) + interleave(train)
    assert len(inputs) == len({r['case_id'] for r in inputs}) == 72
    occurrences = 0
    for row in inputs:
        for path in row['images']:
            with Image.open(path) as im:
                assert getattr(im, 'n_frames', 1) == 1
                im.load(); assert min(im.size) > 0; occurrences += 1
    (ROOT / 'private/inputs.json').write_text(json.dumps(inputs))
    (ROOT / 'private/supervision.json').write_text(json.dumps(gold))
    receipt = {'time': time.time(), 'local_snapshot_cases': len({str(r['case_id']) for r in rows}), 'excluded_previous_cases': 300,
               'source_disjoint_eligible_before_allocation': {s: len(g) for s, g in groups.items()},
               'new_train_cases': 48, 'repeat_development_cases': 24, 'confirmation_reserved_cases': 96,
               'selection_used_labels': False, 'confirmation_gold_accessed': False, 'decoded_image_occurrences': occurrences,
               'max_case_episodes': 104, 'mandatory_case_episodes': 96, 'new_train_confirmation_case_and_known_source_overlap': 0}
    (ROOT / 'DATA_RECEIPT.json').write_text(json.dumps(receipt, indent=2)); print(json.dumps(receipt))


if __name__ == '__main__': main()
