"""A frozen source-derived diagnostic rule, not a clinical validator or learned policy."""
import re
from audit import inspect_report, TEMPORAL


def envelope(text, arm):
    if arm not in {'Advisory', 'Contract'}:
        raise ValueError('Unknown admission arm')
    parsed = inspect_report(text)
    sections = dict(parsed['sections'])
    if arm == 'Contract':
        if parsed['cross_section_literal_flags']:
            sections = {}
        else:
            sections = {name: ' '.join(sentence for sentence in re.split(r'(?<=[.!?])\s+|\n+', value)
                                      if not TEMPORAL.search(sentence))
                        for name, value in sections.items()}
    return {'available_sections': sections,
            'review_flags': {'temporal_markers': parsed['temporal_markers'],
                             'cross_section_literal_flags': parsed['cross_section_literal_flags']},
            'observed_image_count': 1, 'prior_images_or_reports_received': False,
            'temporal_support_from_this_call': False, 'clinical_truth': None,
            'scope': 'Fallible current-image tool observations. Sections generated independently. '
                     'Missing/withheld content is unknown, never a normal or absent finding.'}


def self_check():
    x='FINDINGS:\nThere is a small nodule.\nIMPRESSION:\nStable nodule. No pleural effusion.'
    raw, filtered = envelope(x, 'Advisory'), envelope(x, 'Contract')
    assert raw['review_flags'] == filtered['review_flags']
    assert filtered['available_sections']['findings'] == 'There is a small nodule.'
    assert filtered['available_sections']['impression'] == 'No pleural effusion.'
    conflict='FINDINGS:\nNo pleural effusion.\nIMPRESSION:\nSmall right pleural effusion.'
    assert envelope(conflict, 'Contract')['available_sections'] == {}
    assert envelope(conflict, 'Contract')['clinical_truth'] is None
    negative='FINDINGS:\nNo focal consolidation concerning for pneumonia.\nIMPRESSION:\nNo evidence of pneumonia.'
    assert envelope(negative, 'Contract')['available_sections']


if __name__ == '__main__':
    self_check()
    print('Admission preserves static observations, unknowns, matching flags and negative clauses')
