"""Bounded A0 audit. Private targets stay outside the public repository."""
import collections
import json
import math
import re
import sys
import time
from pathlib import Path

from audit_rules import reference_bindings




def canonical_output(value, cid):
    """Output conventions, not answer-dependent correction of model values."""
    if cid == 'spesi':
        if isinstance(value, str) and value.casefold() in ('low', 'high'):
            return value.casefold()
        if type(value) in (int, float):
            return 'low' if value == 0 else 'high'
        raise ValueError('Unknown sPESI output convention')
    if type(value) in (int, float) and math.isfinite(value):
        return float(value)
    text = str(value).strip()
    if cid in ('mallampati_score', 'asa_physical_status'):
        match = re.fullmatch(r'(?:Class|ASA)\s+([IVX]+|[1-6])', text, re.I)
        if not match:
            raise ValueError('Unknown class output convention')
        token = match[1].upper()
        return float({'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6}.get(token, token))
    # Full-string parsing: allow documented primary unit with an optional
    # parenthesized secondary weight unit, never arbitrary trailing prose.
    match = re.fullmatch(r'([+-]?\d+(?:\.\d+)?)\s*(?:mEq/L|mmHg|msec|ms|kg|mL/min/1\.73\s*m(?:2|²))?(?:\s*\(\d+(?:\.\d+)?\s*(?:lbs|lb)\))?', text, re.I)
    if not match:
        raise ValueError('Unreviewed output serialization')
    return float(match[1])


def run(private, public, base, package, engineering_deadline):
    if time.time() >= engineering_deadline:
        raise RuntimeError('Engineering budget expired before A0')
    private, public, base, package = map(Path, (private, public, base, package))
    PLAN = json.loads((private/'REVIEW_PLAN.private.json').read_text())
    sys.path.insert(0, str(base))
    sys.path.insert(0, str(package))
    from clinical_catalog import CALCS, coverage, identity
    from medharness.scorers.calc import answer_episode_ids
    visible = json.loads((private/'visible_split.json').read_text())
    gold = json.loads((private/'evaluator_gold.json').read_text())
    old = json.loads((private/'D0_SELECTION.private.json').read_text())['items']
    old_keys = {(x['case_id'], x['target_ordinal']) for x in old}
    frozen = private/'A0_REVIEW_SELECTION.private.json'
    if frozen.exists():
        assert json.loads(frozen.read_text())['positions'] == json.loads(json.dumps(PLAN)), 'Do not silently reselect'
    else:
        frozen.write_text(json.dumps({'time': time.time(), 'positions': PLAN,
        'basis': 'visible input sufficiency, declared formula variant and skill coverage; no model performance used',
        'pre_freeze_output_inspection': 'Candidate output types and conventions were inspected during A0; no reference agreement or model scores used to rank candidates.'}, indent=2))
    manifests = []
    for role, positions in PLAN.items():
        for ordinal, (ci, ai, cid) in enumerate(positions):
            c = visible[ci]
            task = gold[c['id']]
            a = task['calculator_answers'][ai]
            ep = answer_episode_ids(task)[ai]
            patients = {r['patient_id'] for r in c['rows'] if r['episode_id'] == ep}
            if len(patients) != 1:
                raise ValueError('Ambiguous target patient')
            patient = next(iter(patients))
            assert (c['id'], ai) not in old_keys
            assert cid in coverage()
            r = {'target_id': f'{role}_{ordinal+1:02d}', 'role': role,
                 'skill': ('L1' if ordinal < 3 else 'L2') if role == 'A' else role,
                 'case_id': c['id'], 'source_group': c['id'], 'case_ordinal': ci,
                 'original_role': c['role'], 'target_ordinal': ai,
                 'calculator_id': cid, 'original_name': a['name'],
                 'episode_id': ep, 'patient_id': patient,
                 'assistance': 'explicit calculator ID, patient, output episode and output convention',
                 'historical_fact_policy': 'same-patient consistent sex may come from other episodes; all measurements and other inputs require target episode',
                 'output_convention': 'low/high risk derived from raw score' if cid == 'spesi' else 'named formula raw scalar or normalized class code',
                 'prior_development_exposure': c['role'].startswith('train'),
                 'prior_model_evaluation_exposure': True,
                 'original_identity_exact': identity(a['name']) == cid,
                 'status': 'pending', 'rows': c['rows']}
            try:
                b = reference_bindings(cid, c['rows'], patient, ep)
                params = {k: x['value'] for k, x in b.items()}
                output = CALCS[cid].calculate(**params).to_dict()
                ref = canonical_output(output['value'], cid)
                expected = canonical_output(a['final_answer'], cid)
                agrees = ref == expected if isinstance(ref, str) else (
                    isinstance(expected, (int, float)) and abs(ref-expected) <= max(.1, abs(expected)*.02))
                r.update(reference_bindings=b, reference_parameters=params,
                         reference_output=output, canonical_reference=ref,
                         original_benchmark_answer=a['final_answer'], canonical_benchmark=expected,
                         reference_agrees=agrees, status='structurally_ready' if agrees else 'output_disagreement')
            except (ValueError, TypeError, KeyError, ZeroDivisionError) as exc:
                r.update(status='blocked', blocker=str(exc))
            manifests.append(r)
    groups = {role: {r['source_group'] for r in manifests if r['role'] == role} for role in PLAN}
    assert all(len(g) >= 3 for g in groups.values())
    assert not (groups['L1'] & groups['L2'] or groups['L1'] & groups['A'] or groups['L2'] & groups['A'])
    assert all(n <= 2 for n in collections.Counter((r['role'], r['source_group']) for r in manifests).values())
    old_groups = {r['case_id'] for r in old}
    assert not groups['A'] & old_groups
    (private/'TARGET_MANIFEST.private.json').write_text(json.dumps(manifests, indent=2))
    counts = collections.Counter(r['status'] for r in manifests)
    ready = counts['structurally_ready'] == 18
    report = {
        'phase': 'A0', 'created_epoch': time.time(),
        'status': 'A0_READY_FOR_INTERFACE' if ready else 'DATA_BLOCKED',
        'source_tasks': len(visible), 'source_outputs': sum(len(t['calculator_answers']) for t in gold.values()),
        'review_targets': len(manifests), 'status_counts': dict(counts),
        'by_role': {role: {'targets': sum(r['role'] == role for r in manifests),
                         'case_groups': len(groups[role]),
                         'ready': sum(r['role'] == role and r['status'] == 'structurally_ready' for r in manifests)} for role in PLAN},
        'selected_old_six': 0, 'role_groups_disjoint': True,
        'anchor_groups_exclude_old_development': True,
        'source_independence': 'Disjoint distributed source-task groups; biological patient independence is not established.',
        'prior_exposure': 'All 32 tasks participated in earlier model evaluation. Original training groups were used for development. Original selection groups reassigned to L1 are now development-only. Anchors remain evaluation-only. No fresh-confirmation claim.',
        'future_hcl_readiness': False,
        'future_confirmation_blocker': '24 suitable targets across at least 12 fresh source groups have not been established; the existing confirmation panel has already been used.',
        'fixture_limitations': ['timestamps and sample_type are generated heuristically, not authentic clinical metadata',
                               'unit strings can truncate multiword units; original field/value text remains the authoritative measurement description'],
        'model_requests': 0, 'real_model_generations': 0,
        'hcl_started': False,
        'targets': [{'role': r['role'], 'skill': r['skill'], 'calculator_id': r['calculator_id'],
                     'status': r['status'], 'blocker': r.get('blocker'),
                     'reference_agrees': r.get('reference_agrees')} for r in manifests],
    }
    public.mkdir(parents=True, exist_ok=True)
    (public/'DATA_READINESS.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return ready
