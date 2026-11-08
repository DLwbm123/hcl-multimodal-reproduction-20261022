"""Focused checks for data isolation, policy parity, and failure denominators."""
import json
from pathlib import Path
from policy import ARMS, FOCUSED, policy_context, summarize


def check(root):
    inputs = json.loads((root/'private/inputs.json').read_text())
    bank = json.loads((root/'private/raw_bank.json').read_text())
    candidate = json.loads((root/'private/candidate.json').read_text())
    assert len(inputs) == 24 and len({x['source_key'] for x in inputs}) == 24
    for item in inputs:
        assert not {'answer', 'gold', 'target'} & item.keys()
        assert len(item['images']) == len(item['image_labels'])
        for arm in ARMS:
            context, ids = policy_context(arm, item['question'], bank, candidate)
            assert len(ids) == (2 if arm == 'Raw' else 0)
            if arm in ['Fixed', 'Proposed_v0']:
                assert FOCUSED in context
    # Missing/failed episodes remain in every arm's planned denominator.
    demo = [{'id': 'one', 'reference_tier': 'source_aligned_not_adjudicated'},
            {'id': 'two', 'reference_tier': 'unresolved'}]
    rows = [{'id': 'one', 'arm': 'Fixed', 'status': 'completed', 'predicted': 'A'},
            {'id': 'one', 'arm': 'Proposed_v0', 'status': 'failed', 'predicted': None}]
    result = summarize(demo, rows, {'one': 'A', 'two': 'B'})
    assert all(m['denominator'] == 2 for m in result['metrics'].values())
    assert result['paired']['all_benchmark_targets']['Fixed_to_Proposed_v0']['loss'] == 1
    try:
        summarize(demo, rows+rows[:1], {'one': 'A', 'two': 'B'})
        raise AssertionError('Duplicate accepted')
    except ValueError:
        pass
    return {'passed': True, 'planned_inputs': len(inputs), 'planned_episodes': len(inputs)*len(ARMS),
            'gold_excluded_from_controller_inputs': True, 'missing_failure_denominator_check': True,
            'duplicate_episode_rejected': True, 'actual_model_execution_tested': False}
