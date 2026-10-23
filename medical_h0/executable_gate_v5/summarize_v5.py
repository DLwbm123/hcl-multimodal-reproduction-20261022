"""Aggregate the frozen scorer and explicit post-run review; never runs a model."""
import collections
import json
from pathlib import Path
from score_v5 import score


def summarize(run, manifest_path, review_path):
    run = Path(run)
    targets = json.loads(Path(manifest_path).read_text())
    reviews = json.loads(Path(review_path).read_text())
    rows, groups = [], collections.defaultdict(set)
    accepted = reviewed = 0
    diagnostics = collections.Counter()
    for target in targets:
        tid = target['target_id']
        path = run/'episodes'/tid/'RESULT.private.json'
        record = json.loads(path.read_text()) if path.exists() else {}
        result = score(record, target)
        manual = reviews.get(tid, {})
        ledger_ids = {r['result_id'] for r in record.get('result_ledger', [])}
        if set(manual) - ledger_ids:
            raise ValueError('Review contains a nonexistent result')
        missing = ledger_ids - set(manual)
        for rid, decision in manual.items():
            assert decision['status'] in ('supported', 'incorrect', 'unverified')
            assert decision['reason'].strip()
        answer = record.get('answer') or {}
        submitted_review = manual.get(answer.get('result_id'), {}).get('status')
        trusted = result['trusted_correct'] and submitted_review == 'supported'
        unresolved = len(missing) + sum(v['status'] == 'unverified' for v in manual.values())
        accepted += len(ledger_ids)
        reviewed += len(manual)
        diagnostics['invalid_tool_calls'] += sum(len(e.get('invalid_tool_calls', [])) for e in record.get('trace', []))
        diagnostics['valid_tool_calls'] += sum(len(e.get('tool_calls', [])) for e in record.get('trace', []))
        diagnostics['tool_error_observations'] += sum(e.get('role') == 'tool' and e.get('status') == 'error' for e in record.get('trace', []))
        calls = [c for e in record.get('trace', []) for c in e.get('invalid_tool_calls', [])]
        error_text = ' '.join(e.get('content', '') for e in record.get('trace', []) if e.get('role') == 'tool' and e.get('status') == 'error')
        diagnostics['targets_with_invalid_tool_call'] += bool(calls)
        diagnostics['targets_with_invalid_json'] += any('Expecting ' in (c.get('error') or '') or 'Unterminated string' in (c.get('error') or '') for c in calls)
        diagnostics['targets_with_scope_or_unread_source_error'] += any(s in error_text for s in ('target-episode evidence', 'Cross-patient', 'must exist and have been read', 'conflicts with explicit target'))
        diagnostics['targets_with_submission_error'] += any(e.get('name') == 'submit_answer' and e.get('status') == 'error' for e in record.get('trace', []))
        if any(a['status'] == 'supported' and manual.get(rid, {}).get('status') == 'supported'
               for rid, a in result['result_audits'].items()) and answer.get('status') != 'success':
            diagnostics['supported_computation_without_successful_submission'] += 1
        if trusted:
            groups[target['role']].add(target['source_group'])
        rows.append({'target': tid, 'role': target['role'], 'skill': target['skill'],
                     'calculator': target['calculator_id'], 'attempted': path.exists(),
                     'termination': record.get('termination', 'not_attempted'),
                     **result, 'trusted_correct': trusted,
                     'manual_unverified_results': unresolved,
                     'manual_incorrect_results': sum(v['status'] == 'incorrect' for v in manual.values()),
                     'model_requests': record.get('model_requests', 0)})
    by_role = {}
    for role in ('L1', 'L2', 'A'):
        subset = [r for r in rows if r['role'] == role]
        by_role[role] = {'targets': len(subset),
                         'numeric_correct': sum(r['numeric_correct'] for r in subset),
                         'trusted_correct': sum(r['trusted_correct'] for r in subset),
                         'trusted_source_groups': len(groups[role])}
    run_result = json.loads((run/'RUN_RESULT.json').read_text()) if (run/'RUN_RESULT.json').exists() else {}
    exit_result = json.loads((run/'EXIT.json').read_text()) if (run/'EXIT.json').exists() else {}
    supervisor = json.loads((run/'SUPERVISOR_EXIT.json').read_text()) if (run/'SUPERVISOR_EXIT.json').exists() else {}
    complete = len(rows) == 18 and all(r['attempted'] for r in rows)
    engineering = complete and run_result.get('status') == 'complete' and exit_result.get('code') == 0 and supervisor.get('code') == 0 and not (run/'HARD_STOP.json').exists()
    unresolved = sum(r['unverified_results'] + r['manual_unverified_results'] for r in rows)
    anchor_skills = {skill: sum(r['trusted_correct'] for r in rows if r['role'] == 'A' and r['skill'] == skill) for skill in ('L1', 'L2')}
    gates = {'all_18_attempted': complete, 'engineering_valid': engineering,
             'L1_min_2_trusted_2_groups': by_role['L1']['trusted_correct'] >= 2 and len(groups['L1']) >= 2,
             'L2_min_2_trusted_2_groups': by_role['L2']['trusted_correct'] >= 2 and len(groups['L2']) >= 2,
             'anchors_each_skill_and_2_groups': min(anchor_skills.values()) >= 1 and len(groups['A']) >= 2,
             'all_accepted_audited': unresolved == 0 and accepted == reviewed}
    if run_result.get('status') == 'BUDGET_STOP' or (run/'HARD_STOP.json').exists():
        status = 'BUDGET_STOP'
    elif unresolved:
        status = 'NEEDS_REVIEW'
    else:
        status = 'PASS' if all(gates.values()) else 'H0_GATE_FAIL'
    return {'status': status, 'gates': gates, 'by_role': by_role,
            'anchor_trusted_by_skill': anchor_skills, 'planned_targets': 18,
            'attempted_targets': sum(r['attempted'] for r in rows),
            'numeric_correct': sum(r['numeric_correct'] for r in rows),
            'trusted_correct': sum(r['trusted_correct'] for r in rows),
            'accepted_computations': accepted, 'manually_reviewed_computations': reviewed,
            'diagnostics': dict(diagnostics),
            'termination_counts': dict(collections.Counter(r['termination'] for r in rows)),
            'state_report_counts': dict(collections.Counter(r['state_report'] for r in rows)),
            'rows': rows, 'hcl_started': False, 'fresh_confirmation': False,
            'scorer_source_commit': '469cde3',
            'aggregation_note': 'Post-run packaging only. Frozen scores cannot be promoted by manual review; review can withhold trust.'}


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('run'); p.add_argument('manifest'); p.add_argument('review'); p.add_argument('output')
    a = p.parse_args()
    Path(a.output).write_text(json.dumps(summarize(a.run, a.manifest, a.review), indent=2)+'\n')
