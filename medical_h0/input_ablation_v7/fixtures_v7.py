"""New synthetic navigation cases plus the six previously unattempted mappings."""
import copy
import bootstrap_v7
from fixtures import row, target, cases as previous_cases


def navigation_cases():
    out = []
    # Entries are (row index, requested field, value, optional scope override).
    specifications = [
        ('first_page', ['marker_a'], [(1, 'marker_a', 'violet', {})]),
        ('middle_page', ['marker_b'], [(6, 'marker_b', 'silver', {})]),
        ('last_page', ['marker_c'], [(10, 'marker_c', 'copper', {})]),
        ('two_fields_split', ['marker_d', 'marker_e'], [(0, 'marker_d', 'north', {}), (9, 'marker_e', 'south', {})]),
        ('three_fields_split', ['marker_f', 'marker_g', 'marker_h'], [(2, 'marker_f', 17, {}), (5, 'marker_g', 29, {}), (11, 'marker_h', 43, {})]),
        ('scope_distractors', ['marker_i'], [(1, 'marker_i', 'decoy', {'patient': 'other_person'}), (4, 'marker_i', 'old', {'episode': 'other_visit'}), (10, 'marker_i', 'current', {})]),
        ('scope_and_two_fields', ['marker_j', 'marker_k'], [(0, 'marker_j', 51, {'episode': 'other_visit'}), (3, 'marker_k', 62, {'patient': 'other_person'}), (5, 'marker_j', 73, {}), (8, 'marker_k', 84, {})]),
        ('consistent_repeat', ['marker_l'], [(0, 'marker_l', 'confirmed', {}), (11, 'marker_l', 'confirmed', {})]),
        ('missing_field', ['marker_m'], []),
        ('missing_target_scope', ['marker_n'], [(2, 'marker_n', 'outside', {'patient': 'other_person'}), (9, 'marker_n', 'outside', {'episode': 'other_visit'})]),
        ('conflict_split', ['marker_o'], [(2, 'marker_o', 'open', {}), (10, 'marker_o', 'closed', {})]),
        ('multi_field_conflict', ['marker_p', 'marker_q'], [(1, 'marker_p', 19, {}), (5, 'marker_q', 23, {}), (9, 'marker_q', 37, {})]),
    ]
    for i, (kind, fields, entries) in enumerate(specifications, 1):
        patient, episode = f'synthetic_person_{i:02d}', f'synthetic_visit_{i:02d}'
        rows = [row(f'irrelevant_{i:02d}_{j:02d}', 700+i*13+j, patient=patient, episode=episode) for j in range(12)]
        for index, field, value, scope in entries:
            rows[index] = row(field, value, patient=scope.get('patient', patient), episode=scope.get('episode', episode))
        t = {**target(), 'patient_id': patient, 'episode_id': episode}
        prompt = ('Find evidence for these exact field names: '+', '.join(fields)+'. '
                  'Report all facts matching these fields in the target patient and episode, including consistent repeated measurements. '
                  'Every requested field must be supported. Preserve other-patient and other-episode records as distractors, not evidence. '
                  'If a requested field is absent after searching the finite input, abstain and explain what is missing. '
                  'If a requested field has conflicting values, abstain and explain the conflict; there is no rule authorizing one conflicting value over another. '
                  'Pages contain at most four facts. Use the available evidence and tools; do not invent facts.')
        out.append({'id':f'C{i:02d}', 'stage':'navigation', 'kind':kind, 'rows':rows, 'target':t,
                    'required_fields':fields, 'prompt':prompt})
    return out


def expected(case):
    """Evaluator only; never added to model inputs or runtime validation."""
    matches = {field:[] for field in case['required_fields']}
    for i, r in enumerate(case['rows'], 1):
        if r['field_name'] in matches and all(r[k] == case['target'][k] for k in ('patient_id','episode_id')):
            matches[r['field_name']].append((f'fact_{i:04d}', r['value_text']))
    missing = [k for k,v in matches.items() if not v]
    conflict = [k for k,v in matches.items() if len({value for _,value in v}) > 1]
    outcome = 'missing' if missing else 'conflict' if conflict else 'answerable'
    return {'outcome':outcome, 'fact_ids':[fid for v in matches.values() for fid,_ in v],
            'conflict_fact_ids':[fid for k in conflict for fid,_ in matches[k]],
            'missing_fields':missing, 'conflicting_fields':conflict}


def schedule():
    mappings = [copy.deepcopy(c) for c in previous_cases() if c['stage']=='mapping']
    scheduled = []
    for i, case in enumerate(navigation_cases()):
        for arm in (['A','B'] if i % 2 == 0 else ['B','A']):
            scheduled.append({**copy.deepcopy(case), 'case_id':case['id'], 'id':case['id']+'_'+arm, 'arm':arm})
        if i % 2 == 1:
            m = mappings[i//2]
            scheduled.append({**m, 'case_id':m['id'], 'id':m['id']+'_B', 'arm':'B'})
    return scheduled
