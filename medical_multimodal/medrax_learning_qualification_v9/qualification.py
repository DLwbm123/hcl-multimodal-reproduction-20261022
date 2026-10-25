"""Frozen, deterministic error-source qualification; invalid runs are not sources."""
def valid(row):
    return row.get('status') == 'completed' and row.get('predicted') in list('ABCDEF')


def first_errors(items, predictions, gold, maximum=4):
    by = {r['id']: r for r in predictions}
    return [x for x in items if x['id'] in by and valid(by[x['id']]) and by[x['id']]['predicted'] != gold[x['id']]][:maximum]


def source_summary(items, initial, rechecks, selected, gold):
    before = {r['id']: r for r in initial}; after = {r['id']: r for r in rechecks}
    summary = {'training_denominator': len(items), 'training_valid': 0, 'training_correct': 0, 'training_valid_errors': 0,
               'training_missing_or_invalid': 0, 'selected_for_recheck': len(selected), 'recheck_valid': 0,
               'wrong_twice': 0, 'same_wrong_option': 0, 'different_wrong_option': 0, 'recovered': 0, 'recheck_missing_or_invalid': 0}
    for x in items:
        row = before.get(x['id'], {})
        if valid(row):
            summary['training_valid'] += 1
            summary['training_correct' if row['predicted'] == gold[x['id']] else 'training_valid_errors'] += 1
        else: summary['training_missing_or_invalid'] += 1
    for x in selected:
        row = after.get(x['id'], {})
        if not valid(row): summary['recheck_missing_or_invalid'] += 1; continue
        summary['recheck_valid'] += 1
        if row['predicted'] == gold[x['id']]: summary['recovered'] += 1
        else:
            summary['wrong_twice'] += 1
            summary['same_wrong_option' if row['predicted'] == before[x['id']]['predicted'] else 'different_wrong_option'] += 1
    summary['preliminary_one_bank_sources'] = summary['wrong_twice'] >= 2
    summary['preliminary_two_disjoint_banks_sources'] = summary['wrong_twice'] >= 4
    return summary
