"""Aggregate a completed native integration without publishing case-level evidence."""
from collections import Counter
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent


def read(name):
    return json.loads((ROOT / name).read_text())


def main():
    result = read('FINISHED.json')
    episodes = read('private/episodes.json')
    metadata = read('private/hcl_calls.json')
    events = [json.loads(x) for x in (ROOT/'private/events.jsonl').read_text().splitlines()]
    finished = [e for e in events if e['kind'] == 'controller_finished']
    keys = [e['key'] for e in finished]
    assert len(keys) == len(set(keys)) == result['controller_calls'] == 146
    assert len(episodes) == result['native_episodes'] == result['read_only_checks'] == 46
    assert all(x['status'] == 'completed' and x['predicted'] in 'ABCDEF' for x in episodes)
    states = [read('private/states/'+n+'.json') for n in ['S0', 'SA', 'SAB']]
    assert states[0] == states[1] == states[2]
    assert result['engineering_loop_complete'] and result['accepted_updates'] == 0
    assert set(result['decisions']) == {'A', 'B'}
    assert all(len(d['candidates']) == 2 for d in result['decisions'].values())
    assert all(c['anchor_denominator'] == c['anchor_protected'] == 2 and c['anchor_losses'] == 0
               for c in result['decisions']['B']['candidates'])
    assert all(c['current_denominator'] == c['before_correct'] == c['after_correct'] == 4
               for d in result['decisions'].values() for c in d['candidates'])
    assert read('EXIT.json')['worker_exit_code'] == 0
    totals = Counter()
    for event in finished:
        assert event['status'] == 'ok'
        assert len(event.get('usage', [])) == 1
        for k,v in event['usage'][0].items():
            if isinstance(v, (float, int)):
                totals[k] += v
    qualifications = [e for e in events if e['kind'] == 'expert_qualification']
    assert len(qualifications) == 35 and all(e['visual_contract_pass'] for e in qualifications)
    routes = [e for e in events if e['kind'] == 'hcl_route' and e.get('selected_memory')]
    assert len(routes) == 20
    costs = {
        'controller_requests': len(finished), 'native_controller_requests': len(finished)-len(metadata),
        'hcl_metadata_requests': len(metadata),
        'proposal_requests': sum(x.get('phase') == 'proposal' for x in metadata),
        'routing_requests': sum(x.get('phase') == 'episode' for x in metadata),
        'controller_usage': dict(totals),
        'usage_note': 'Cached input is a subset of input; reasoning output is a subset of output. Do not add these twice.',
        'tool_calls': dict(Counter(e['name'] for e in events if e['kind'] == 'tool_started')),
        'tool_generation_calls': sum(e['kind'] == 'expert_generation_finished' for e in events),
        'tool_output_tokens': sum(e['generated_sequence_length'] for e in events if e['kind'] == 'expert_generation_finished'),
        'tool_token_note': 'Separate local tool-model output count; not part of subscription controller usage.',
        'episodes_with_tool_calls': sum(x['tool_calls'] > 0 for x in episodes),
        'seconds_from_original_start': result['seconds_since_original_start'],
        'cash_cost': None}
    audit = {
        'engineering_complete': True, 'all_46_episodes_valid': True,
        'four_candidates_really_evaluated': True, 'both_B_candidates_used_two_A_anchors': True,
        'read_only_checks': 46, 'nonempty_candidate_memory_selections': len(routes),
        'all_candidate_evaluations_restored_incoming': all(c['restored_incoming_after_evaluation']
            for d in result['decisions'].values() for c in d['candidates']),
        'S0_SA_SAB_contents_equal': True, 'accepted_updates': 0,
        'acquired_A_retention_is_null': result['retention']['acquired_A_retention'] is None,
        'expert_visual_encoding_passes': len(qualifications), 'worker_exit_code': 0,
        'clinical_validity_established': False, 'learning_benefit_established': False,
        'report_actual_unique_execution_arms': ['H0', 'Fixed'],
        'report_state_aliases': result['state_aliases'],
        'source_scope': '24 previously exposed development sources; protected sets unused'}
    for name,value in [('COST_REPORT.json',costs),('DELIVERY_AUDIT.json',audit)]:
        (ROOT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    # Keep a complete trace's locator and event sequence private.
    for route in routes:
        chosen=[e for e in events if e.get('episode_id') == route['episode_id'] and e.get('arm') == route['arm']]
        if any(e['kind']=='tool_started' for e in chosen):
            row=next(x for x in episodes if x['id']==route['episode_id'] and x['arm']==route['arm'])
            trace={'episode':row,'events':chosen,'parent':'S0' if row['stage']=='A' else 'SA',
                   'candidate_decision':result['decisions'][row['stage']],
                   'warning':'PRIVATE: contains input context and model/tool observations; never publish.'}
            (ROOT/'private/new_hcl_trace.json').write_text(json.dumps(trace,ensure_ascii=False,indent=2)+'\n')
            break
    print(json.dumps({'costs':costs,'audit':audit},ensure_ascii=False))


if __name__ == '__main__':
    main()
