"""Post-hoc provenance audit of existing paired report observations; no clinical scores."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

TEMPORAL = re.compile(r'\b(stable|unchanged|interval|increased|decreased|improved|worsen\w*|resolv\w*)\b', re.I)
SECTIONS = re.compile(r'\AFINDINGS:\s*(.*?)\s*IMPRESSION:\s*(.*)\Z', re.S)
# These narrow literal checks are review flags, not a medical contradiction classifier.
OPPOSED = {
    'pleural_effusion': (r'\bno (?:evidence of |definite )?pleural effusions?\b',
                        r'\b(?:small|moderate|large|trace) (?:(?:left|right|bilateral)[ -](?:sided )?)?pleural effusions?\b'),
    'pneumonia': (r'\bno evidence of pneumonia\b', r'\bconcerning for pneumonia\b'),
}


def positive_mention(pattern, text):
    # Conservative sentence scope: a negated positive phrase is not a positive finding.
    for sentence in re.split(r'[.!?\n]+', text):
        match = re.search(pattern, sentence, re.I)
        if match and not re.search(r'\b(no|without|not|absent|negative)\b', sentence[:match.end()], re.I):
            return True
    return False


def inspect_report(text):
    prefix = 'CHEST X-RAY REPORT\n\n'
    match = SECTIONS.fullmatch(text[len(prefix):] if text.startswith(prefix) else text)
    if match is None:
        raise ValueError('Unrecognized report structure; do not silently admit')
    findings, impression = match.groups()
    flags = []
    for concept, (negative, positive) in OPPOSED.items():
        if ((re.search(negative, findings, re.I) and positive_mention(positive, impression)) or
            (re.search(negative, impression, re.I) and positive_mention(positive, findings))):
            flags.append(concept)
    return {'sections': {'findings': findings, 'impression': impression},
            'temporal_markers': sorted({m.group().lower() for m in TEMPORAL.finditer(text)}),
            'cross_section_literal_flags': flags,
            'clinical_truth': None,
            'temporal_support_from_this_call': False,
            'provenance_reason': 'Generator receives one image and no comparison history; sections are independently generated.'}


def audit(inputs, events, annotations):
    known = {x['id']: x for x in inputs}
    if len(known) != 46 or set(known) != set(annotations):
        raise ValueError('Expected the complete 46-source learning registry')
    rows = []
    seen = set()
    for event in events:
        if event['kind'] != 'paired_probe':
            continue
        key = event['episode_id']
        if key not in known or key in seen or event.get('arm') != 'Probe':
            raise ValueError('Invalid or duplicate source provenance')
        seen.add(key)
        obs = event['observations']
        if [x['image_path'] for x in obs] != known[key]['tool_eligible_images']:
            raise ValueError('Observation/image qualification mismatch')
        for index, item in enumerate(obs):
            if item['status'] != 'completed':
                raise ValueError('Noncompleted observations must be handled separately')
            rows.append({'id': key, 'observation_index': index, 'image_path': item['image_path'],
                         'raw_observation': item['observation'], **inspect_report(item['observation'])})
    expected = {x['id'] for x in inputs if x['tool_eligible_images']}
    if seen != expected or len(rows) != sum(len(x['tool_eligible_images']) for x in inputs):
        raise ValueError('Incomplete report coverage')
    temporal = [x for x in rows if x['temporal_markers']]
    flagged = [x for x in rows if x['cross_section_literal_flags']]
    summary = {
        'analysis_type': 'post_hoc_source_learning_audit',
        'new_model_calls': 0,
        'source_groups': len(inputs), 'sources_with_reports': len(seen),
        'sources_without_eligible_report_images': len(inputs)-len(seen),
        'report_observations': len(rows),
        'source_semantic_eligibility': dict(Counter('qualified' if x['semantic_feedback_allowed'] else 'unknown' for x in inputs)),
        'manual_primary_demand_counts': dict(Counter(x['primary_demand'] for x in annotations.values())),
        'temporal_marker_reports': len(temporal),
        'sources_with_temporal_marker_reports': len({x['id'] for x in temporal}),
        'cross_section_literal_flag_reports': len(flagged),
        'sources_with_cross_section_literal_flags': len({x['id'] for x in flagged}),
        'literal_flag_counts': dict(Counter(f for x in flagged for f in x['cross_section_literal_flags'])),
        'clinical_accuracy_measured': False,
        'contradiction_detector_validated': False,
        'manual_demand_labels_are_provisional_single_reviewer': True,
        'evaluation_or_confirmation_read': False,
        'unknown_clinical_truth_preserved': all(x['clinical_truth'] is None for x in rows),
        'candidate_accepted': False,
    }
    return rows, summary


def self_check():
    text = 'CHEST X-RAY REPORT\n\nFINDINGS:\nNo pleural effusion.\n\nIMPRESSION:\nSmall left pleural effusion, unchanged.'
    result = inspect_report(text)
    assert result['cross_section_literal_flags'] == ['pleural_effusion']
    assert result['temporal_markers'] == ['unchanged']
    assert result['clinical_truth'] is None and not result['temporal_support_from_this_call']
    clean = inspect_report('FINDINGS:\nThere is no evidence of pneumonia.\nIMPRESSION:\nNo acute findings.')
    assert not clean['cross_section_literal_flags'] and not clean['temporal_markers']
    negated = inspect_report('FINDINGS:\nNo focal consolidations concerning for pneumonia.\nIMPRESSION:\nNo evidence of pneumonia.')
    assert not negated['cross_section_literal_flags']
    try:
        inspect_report('Error generating report')
    except ValueError:
        pass
    else:
        raise AssertionError('Malformed observations must fail closed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--events', type=Path)
    parser.add_argument('--annotations', type=Path)
    parser.add_argument('--private-output', type=Path)
    parser.add_argument('--summary', type=Path)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
        print('Synthetic provenance and parser checks passed')
    else:
        rows, summary = audit(json.loads(args.inputs.read_text()),
                              [json.loads(x) for x in args.events.read_text().splitlines()],
                              json.loads(args.annotations.read_text()))
        args.private_output.write_text(json.dumps(rows, indent=2) + '\n')
        args.summary.write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary, indent=2))
