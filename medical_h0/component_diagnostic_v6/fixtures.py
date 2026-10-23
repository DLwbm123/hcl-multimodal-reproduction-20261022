"""Predeclared synthetic component fixtures. No benchmark records or answers."""

def row(field, value, episode='synthetic_visit', patient='synthetic_patient'):
    return {'patient_id': patient, 'episode_id': episode, 'field_name': field,
            'value_text': str(value), 'unit': '', 'observed_at': 'synthetic',
            'sample_type': 'synthetic', 'source': 'synthetic_fixture'}


def target(cid='anion_gap', output='raw'):
    return {'calculator_id': cid, 'patient_id': 'synthetic_patient',
            'episode_id': 'synthetic_visit', 'output_type': output}


def result(rid, value, output='raw', cid='anion_gap'):
    return {**target(cid, output), 'result_id': rid, 'name': cid, 'value': value,
            'unit': 'synthetic', 'formula_version': 'synthetic_submission_fixture'}


def cases():
    out = []
    specs = [
        ('S01', [result('result_001', 12)], 'result_001', 'Submit the completed result result_001.'),
        ('S02', [result('result_001', 2, 'raw', 'spesi'), result('result_002', 'high', 'risk_category', 'spesi')], 'result_002', 'The requested output is a risk category. Submit the existing category result; the raw score is not the requested output.'),
        ('S03', [result('result_017', -2, cid='rass')], 'result_017', 'Submit the completed result result_017, preserving its value.'),
        ('S04', [], None, 'The necessary source measurement is absent and no result exists. End with an explicit abstention; do not invent a result.'),
        ('S05', [], None, 'The available readings are contradictory, and no verified result exists. Explicitly abstain with a reason.'),
        ('S06', [result('result_001', 8), result('result_002', 14)], 'result_002', 'Two immutable results are available. The task explicitly requests result_002. Submit that result without changing either result.'),
    ]
    for cid, ledger, expected, prompt in specs:
        t = {k: ledger[-1][k] for k in target()} if ledger else target()
        out.append({'id': cid, 'stage': 'submission', 'rows': [], 'target': t,
                    'seed_results': ledger, 'expected_result_id': expected, 'prompt': prompt})
    for n in range(1, 7):
        rows = [row('distractor_'+str(i), 100+i) for i in range(12)]
        field = 'requested_marker'
        expected, seed = [9], None
        rows[9] = row(field, 'present')
        prompt = 'Read the synthetic facts and report only the fact IDs for requested_marker in the target patient and episode. Pages contain at most four facts.'
        if n == 2:
            rows[1] = row('first_marker', 7); rows[10] = row('second_marker', 9)
            expected = [1, 10]
            prompt = 'Report the two fact IDs supporting first_marker and second_marker for the target scope. Read the needed pages, at most four facts per page.'
        elif n == 3:
            rows[0] = row(field, 'old', episode='old_visit')
            rows[2] = row(field, 'other', patient='other_synthetic_patient')
            rows[9] = row(field, 'current')
        elif n == 4:
            rows[1] = row('saved_marker', 'available'); expected = [1]; seed = 'read_first'
            prompt = 'Prior interaction history is no longer displayed. Find saved_marker in the persistent evidence state and report its supporting fact ID for the target scope.'
        elif n == 5:
            rows[2] = row('recover_marker', 'available'); expected = [2]; seed = 'read_first'
            prompt = 'First retrieve observation_001 with get_observation, then report the fact ID for recover_marker in that observation. This explicitly tests observation recovery.'
        elif n == 6:
            seed = 'unread_failure'
            prompt += ' A previous report was rejected because its source had not been read. Read the needed evidence, then retry the report.'
        out.append({'id': f'N{n:02d}', 'stage': 'navigation', 'rows': rows, 'target': target(),
                    'seed': seed, 'expected_fact_ids': [f'fact_{i+1:04d}' for i in expected],
                    'require_get_observation': n == 5, 'prompt': prompt})
    mappings = [
        ('M01', 'anion_gap', [row('Sodium', '142 mEq/L'), row('Chloride', '104 mEq/L'), row('Bicarbonate', '24 mEq/L')]),
        ('M02', 'anion_gap', [row('Sodium', '138 mEq/L'), row('Chloride', '101 mEq/L'), row('Bicarbonate', '22 mEq/L')]),
        ('M03', 'mme_calculator', [row('Drug', 'Oxycodone'), row('Dosage (mg/dose)', 5), row('Doses per day (doses/day)', 3), row('Add another drug', 'No')]),
        ('M04', 'mme_calculator', [row('Drug', 'Oxycodone'), row('Dosage (mg/dose)', 10), row('Doses per day (doses/day)', 2), row('Add another drug', 'No')]),
        ('M05', 'rass', [row('patient_description', 'Restless (+1)')]),
        ('M06', 'rass', [row('patient_description', 'Alert and calm (0)')]),
    ]
    for tid, cid, rows in mappings:
        out.append({'id': tid, 'stage': 'mapping', 'rows': rows, 'target': target(cid),
                    'prompt': f'For the synthetic patient and episode, read the facts and formula definition, bind supported parameters, execute {cid}, and explicitly submit the immutable result. Use only the stated facts.'})
    return out
