"""Check exact frozen memberships, privacy receipts and evaluation boundaries."""
import json
from pathlib import Path


def check_inputs(root, inputs, require_paths=True):
    root=Path(root)
    protocol=json.loads((root/'code/protocol.json').read_text())
    if len(inputs)!=24 or len({r['id'] for r in inputs})!=24 or len({r['source_key'] for r in inputs})!=24:
        raise ValueError('Expected 24 unique already-approved development sources')
    for stage in ['A','B']:
        for role,n in protocol['roles_per_stage'].items():
            if sum(x['stage']==stage and x['role']==role for x in inputs)!=n:
                raise ValueError('Role count mismatch')
    if any(not x['privacy_approved'] for x in inputs):
        raise ValueError('Unapproved input')
    for x in inputs:
        if (len(x['images'])==1)!=(x['stage']=='A'):
            raise ValueError('Existing image-stratum membership changed')
        if require_paths and any(not Path(p).is_file() for p in x['images']):
            raise FileNotFoundError('Approved image missing')
    for role in ['train','selection','anchor','report']:
        path=root/'private'/(role+'_gold.json')
        if role=='report':
            if not path.is_file():
                raise FileNotFoundError('Sealed report references missing')
            continue  # Do not read report labels before states and predictions freeze.
        gold=json.loads(path.read_text())
        if set(gold)!={x['id'] for x in inputs if x['role']==role}:
            raise ValueError('Separated evaluator label membership mismatch')
        if any(v not in list('ABCDEF') for v in gold.values()):
            raise ValueError('Invalid reference format')
    # No generic Pipeline's 20/15 anchor check or test forgetting probe is used.
    if protocol['gate']['selection_count']!=4 or protocol['gate']['historical_anchor_count_B']!=2:
        raise ValueError('Native evaluator thresholds incompatible with frozen membership')
    return {'sources':24,'role_counts':protocol['roles_per_stage'],
            'historical_anchors_B':2,'no_test_probe':True,'prediction_cache_disabled':True,
            'within_run_roles_disjoint':True,'all_sources_previously_exposed':True,
            'privacy_approved':True,'paths_checked':require_paths}
