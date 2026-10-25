"""Offline endpoint audit; no model calls or changes to the frozen experiment."""
import ast
import collections
import json
import os
from pathlib import Path
import re
from notes_logic import ARMS, parse_note


def main():
    root = Path(os.environ['R981_ROOT'])
    read = lambda name: json.loads((root / name).read_text())
    finish = read('FINISHED.json')
    episodes = read('private/episodes.json')
    inputs = read('private/inputs.json')
    gold = read('private/sealed_report_gold.json')
    states = read('private/states.json')
    source = read('private/source_state.json')
    notes = read('private/notes.json')
    results = read('private/report_results.json')
    assert finish['fatal'] is None and len(episodes) == 144
    assert all(x['status'] == 'completed' and x['predicted'] in 'ABCDEF' for x in episodes)
    assert len(inputs) == len({x['case_id'] for x in inputs}) == len(gold) == 48
    assert all(set(x) == {'id', 'case_id', 'stage', 'role', 'question', 'images'} and x['role'] == 'report' for x in inputs)
    report_ids = {x['id'] for x in inputs}
    train_ids = {x['internal_source']['task_id'] for x in source['raw']}
    assert len(train_ids) == 2 and not train_ids & report_ids
    original = json.loads((Path(os.environ['R1002_SOURCE']) / 'private/states.json').read_text())['corrective_memory']
    assert source == original == states['raw_errors']
    assert states['error_notes']['abstract'] == notes and len(notes) == 2
    for i, note in enumerate(notes):
        generation = read('private/note_generation_' + str(i) + '.json')
        assert parse_note(generation['raw_output']) == note['note']
        assert not generation['tool_calls'] and generation['source_id'] in train_ids and generation['images'] > 0
    from hcl.memory import ExperienceMemory, MemoryConfig
    from worker import context
    from logic import paired
    memory = ExperienceMemory(MemoryConfig(record_dir=str(root / 'private/audit_memory'), raw_capacity_per_task=4, abstract_capacity_per_task=2, skill_enabled=False, anchor_selection_mode='off'))
    items = {x['id']: x for x in inputs}
    for row in episodes:
        snapshot = states[row['state']]
        memory.restore(snapshot)
        text, ids = context(memory, items[row['id']]['question'], train_ids)
        assert ids == row['retrieved_ids'] and len(text) == row['context_chars'] and memory.snapshot() == snapshot
    per_arm = {}
    for name in ARMS:
        rows = results[name]
        assert len(rows) == 48 and {x['id'] for x in rows} == report_ids
        assert all(x['correct'] == (x['predicted'] == gold[x['id']]) for x in rows)
        for stage in ['A', 'B']:
            assert sum(x['correct'] for x in rows if x['stage'] == stage) == finish['metrics'][name][stage]['correct']
        per_arm[name] = {'records': len(states[name]['raw']) + len(states[name]['abstract']), **finish['metrics'][name]['context']}
    for key, values in finish['paired'].items():
        a, b = key.split(' -> ')
        for stage in ['A', 'B', 'pooled']:
            ids = [x['id'] for x in inputs if stage == 'pooled' or x['stage'] == stage]
            assert paired(results[a], results[b], ids, gold) == values[stage]
    events = [json.loads(x) for x in (root / 'private/events.jsonl').read_text().splitlines()]
    starts = [x for x in events if x['kind'] == 'controller_started']
    calls = [x for x in events if x['kind'] == 'controller_finished']
    tools = [x for x in events if x['kind'] == 'tool_finished']
    assert len(starts) == len(calls) == 447 and all(x['status'] == 'ok' for x in calls)
    assert all(x['images'] > 0 for x in starts)
    note_calls = [x for x in calls if x['state'] == 'note_generation']
    assert len(note_calls) == 2 and min(x['time'] for x in starts) > read('FROZEN.json')['time']
    frozen = read('STATES_FROZEN.json')['time']
    assert max(x['time'] for x in note_calls) < frozen < min(x['time'] for x in starts if x['state'] in ARMS)
    usage = collections.Counter()
    for x in calls:
        for value in x.get('usage', []):
            usage.update(value)
    flags = collections.Counter()
    for x in tools:
        value = ast.literal_eval(x['result'])[0]
        text = value.get('response', '') if isinstance(value, dict) and 'response' in value else value if isinstance(value, str) else None
        if text is None:
            continue
        if not text.strip(): flags['empty'] += 1
        if text.strip().lower() == 'not matched': flags['not_matched'] += 1
        if text.strip().lower() == 'no': flags['bare_no'] += 1
        if 'bibibibibibi' in text: flags['repetitive_malformed'] += 1
    prior = Path(os.environ['R1002_SOURCE'])
    train = {x['id']: x for x in json.loads((prior / 'private/episodes.json').read_text()) if x['role'] == 'train'}
    source_flips = []
    for key in sorted(train_ids):
        row = train[key]
        match = re.match(r'([A-F])(?:[).:\s]|$)', row['initial_final'].lstrip().lstrip('*'))
        source_flips.append(None if match is None else match.group(1) != row['predicted'])
    result = {
        'complete_audit_passed': True, 'physical_report_episodes': 144, 'unique_report_cases': 48,
        'all_contexts_reconstructed': 144, 'source_exactly_matches_frozen_v7_errors': True,
        'notes_exactly_match_original_generation': True, 'source_report_ids_disjoint': True,
        'all_report_calls_after_states_frozen': True, 'all_states_immutable': True,
        'all_paired_tables_recomputed': True, 'per_arm': per_arm,
        'controller_calls': len(calls), 'controller_calls_with_images': len(starts),
        'note_generation_calls': 2, 'note_generation_seconds': sum(x['seconds'] for x in note_calls),
        'controller_usage': dict(usage), 'controller_request_seconds_sum': sum(x.get('seconds', 0) for x in calls),
        'tool_counts': dict(collections.Counter(x['name'] for x in tools)),
        'tool_statuses': dict(collections.Counter(x['status'] for x in tools)),
        'tool_output_surface_flags': dict(flags), 'surface_flags_are_not_medical_correctness_labels': True,
        'source_errors_initial_to_final_option_changed': source_flips,
        'seconds_since_original_start': finish['seconds_since_original_start'], 'final_guard': read('RUN_GUARD.json'),
        'B_gain_threshold_mathematically_reachable': max(finish['metrics'][x]['B']['correct'] for x in ['H0_static', 'raw_errors']) <= 22,
        'development_signal': False, 'notes_medically_validated': False,
        'limitations': ['Zero B gains or losses against either control; A loses one baseline-correct case',
                        'Two fixed B sources and one generation each, without independent source or stochastic replication',
                        'All cases receive both records; applicability text is not a tested retrieval or abstention mechanism',
                        'No source-image ablation; content, length and extra generation differ from raw',
                        'Structural and provenance checks do not establish medical correctness',
                        'No A acquisition or sequential learning and retention established']}
    (root / 'AUDIT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
