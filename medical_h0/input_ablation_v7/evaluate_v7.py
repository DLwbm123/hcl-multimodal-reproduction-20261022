"""Post-episode diagnostics; expected rules never enter the model context."""
import bootstrap_v7
from fixtures_v7 import expected
from run_components import evaluate as previous_evaluate


def evaluate(record, case):
    if case['stage']=='mapping':
        result = previous_evaluate(record, case)
        result['preliminary_correct'] = result.pop('correct')
        result['semantic_review'] = 'pending_all_accepted_computations'
        return result
    truth = expected(case)
    state = record['execution_state']['evidence_state']
    read_ids = {r['fact_id'] for r in state['read_facts']}
    answer = record.get('answer') or {}
    success = answer.get('status')=='success'
    abstain = answer.get('status')=='abstain'
    full = len(read_ids)==len(case['rows'])
    required = set(truth['conflict_fact_ids'] if truth['outcome']=='conflict' else truth['fact_ids'])
    visible = required <= read_ids if truth['outcome']!='missing' else full
    exact = success and set(answer.get('fact_ids',[]))==set(truth['fact_ids']) and truth['outcome']=='answerable'
    structural = exact if truth['outcome']=='answerable' else abstain and visible
    errors = [e for e in record['trace'] if e['role']=='tool' and e.get('status')=='error']
    return {'expected_outcome':truth['outcome'], 'expected_fact_ids':truth['fact_ids'],
            'required_evidence_visible_at_end':visible, 'exact_evidence_selection':exact,
            'all_input_read':full, 'read_fact_count':len(read_ids),
            'ended_with_unread_facts':bool(answer) and not full,
            'incorrect_termination_with_unread_facts':bool(answer) and not full and not structural,
            'abstained':abstain, 'preliminary_correct':structural,
            'scope_error_calls':sum('Evidence outside the target scope' in e['content'] for e in errors),
            'semantic_review':'pending_abstention_reason' if abstain else 'exact_selection_checked'}
