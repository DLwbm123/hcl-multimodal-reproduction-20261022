"""Validate structure only; generated medical statements remain unverified."""
import json
import re

ARMS=['H0_static','raw_errors','error_notes']
FIELDS=['evidence_from_training','wrong_inference','corrective_rule','applicable_when','do_not_apply_when','uncertainty']


def parse_note(text):
    value=json.loads(text)
    if not isinstance(value,dict) or set(value)!=set(FIELDS):raise ValueError('Wrong error-note fields')
    if any(not isinstance(value[k],str) or not value[k].strip() for k in FIELDS):raise ValueError('Empty or invalid note field')
    if sum(len(value[k]) for k in FIELDS)>1800:raise ValueError('Error note exceeds frozen size')
    joined=' '.join(value.values())
    if re.search(r'\b(?:return only|respond only|output only|only respond|option\s+[A-F]\b|choice\s+[A-F]\b)',joined,re.I):raise ValueError('Answer-output directive or historical option label in note')
    return value
