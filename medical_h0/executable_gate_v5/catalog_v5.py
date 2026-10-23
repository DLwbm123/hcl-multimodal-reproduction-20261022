"""Fixed, named formula profiles. No patient data or evaluator imports."""
import copy
import math
import sys
from pathlib import Path
from jsonschema import Draft202012Validator

BASE = Path(__file__).resolve().parent.parent/'interface_diagnostic_v4'
if (BASE/'code'/'clinical_catalog.py').is_file():
    BASE = BASE/'code'
sys.path.append(str(BASE))
import clinical_catalog as original

PARAMETERS = {
    'anion_gap': ['sodium', 'chloride', 'bicarbonate'],
    'winters_formula': ['hco3'],
    'mme_calculator': ['opioid_name', 'daily_dose_mg'],
    'sofa_score': ['pao2_fio2_ratio', 'platelets', 'bilirubin', 'gcs_score', 'creatinine',
                   'map_value', 'dopamine_dose', 'dobutamine_any', 'epinephrine_dose',
                   'norepinephrine_dose', 'is_mechanically_ventilated'],
    'ich_score': ['gcs_score', 'ich_volume_ml', 'ivh_present', 'infratentorial', 'age'],
    'qsofa_score': ['respiratory_rate', 'systolic_bp', 'altered_mentation', 'gcs_score'],
    'rass': ['rass_score'],
    'iss': ['head_neck_ais', 'face_ais', 'chest_ais', 'abdomen_ais', 'extremity_ais', 'external_ais'],
    'perc_rule': ['age_50_or_older', 'heart_rate_100_or_higher', 'o2_sat_below_95',
                  'unilateral_leg_swelling', 'hemoptysis', 'recent_surgery_trauma', 'prior_pe_dvt', 'hormone_use'],
    'spesi': ['age', 'cancer', 'chronic_cardiopulmonary_disease', 'heart_rate', 'systolic_bp', 'spo2'],
    'ideal_body_weight': ['height_cm', 'sex'],
    'corrected_qt': ['qt_interval', 'heart_rate', 'sex', 'formula'],
    'ckd_epi_2021': ['age', 'sex', 'serum_creatinine'],
    'mallampati_score': ['mallampati_class'],
    'asa_physical_status': ['asa_class', 'is_emergency'],
    'rcri': ['high_risk_surgery', 'ischemic_heart_disease', 'heart_failure',
             'cerebrovascular_disease', 'insulin_diabetes', 'creatinine_above_2'],
}

PROFILES = {
    'anion_gap': 'Uncorrected serum anion gap without potassium. Albumin correction is outside this named output contract. Omitted library options retain their documented algorithm defaults; no patient measurement is imputed.',
    'winters_formula': 'Expected PaCO2 center from bicarbonate. Comparison against measured PaCO2 is not requested and no measured PaCO2 is imputed.',
    'mme_calculator': 'Single oral opioid MME/day. Supply total daily dose, not dose per administration. Conditional patch, methadone and custom-factor branches are outside this profile.',
    'ideal_body_weight': 'Devine ideal body weight only. Actual-weight-dependent adjusted body weight is not requested; actual weight is not imputed.',
    'spesi': 'Raw-vital-input sPESI profile. The raw result is a score; if the task requests low/high risk, explicitly call derive_output on that result before submitting.',
    'sofa_score': 'Creatinine-based renal component; urine-output alternative is outside this profile. Explicit treatment status is required, including zero drug doses only where evidence establishes absence.',
}


def details(cid):
    if cid not in PARAMETERS:
        raise ValueError('Calculator is outside this frozen subdomain')
    d = copy.deepcopy(original.details(cid))
    props = {k: d['parameters']['properties'][k] for k in PARAMETERS[cid]}
    # These profiles do not admit null placeholders for missing measurements.
    for k, p in props.items():
        p.pop('upstream_default_not_assumed', None)
        if 'anyOf' in p:
            nonnull = [x for x in p['anyOf'] if x.get('type') != 'null']
            if len(nonnull) == 1:
                props[k] = {**nonnull[0], **{a: b for a, b in p.items() if a != 'anyOf'}}
    if cid == 'mme_calculator':
        props['opioid_name'] = {'enum': ['morphine', 'codeine', 'hydrocodone', 'oxycodone',
            'oxymorphone', 'hydromorphone', 'tramadol', 'tapentadol', 'meperidine',
            'levorphanol', 'pentazocine', 'butorphanol', 'nalbuphine', 'dihydrocodeine']}
    if cid == 'iss':
        for p in props.values():
            p.update(minimum=0, maximum=6)
    if cid in ('sofa_score', 'qsofa_score', 'ich_score') and 'gcs_score' in props:
        props['gcs_score'].update(minimum=3, maximum=15)
    d['parameters'] = {'type': 'object', 'properties': props, 'required': list(props), 'additionalProperties': False}
    d['profile'] = PROFILES.get(cid, 'Explicit inputs to the pinned named formula.')
    d['source_policy'] = 'Measurements and clinical states require target episode. Consistent same-patient Sex may be read from another episode for corrected_qt. Other cross-episode evidence is rejected. Source semantic support is evaluated separately.'
    d['formula_version'] = 'medical-calc-mcp-45f648d9; executable_gate_v5 profiles v1; formula bodies unchanged'
    return d


def execute(cid, params):
    d = details(cid)
    Draft202012Validator(d['parameters']).validate(params)
    for value in params.values():
        if type(value) is float and not math.isfinite(value):
            raise ValueError('Nonfinite input')
    result = original.CALCS[cid].calculate(**params).to_dict()
    if type(result['value']) is float and not math.isfinite(result['value']):
        raise ValueError('Nonfinite formula output')
    return {'calculator_id': cid, 'name': original.CALCS[cid].name,
            'value': result['value'], 'unit': result['unit'],
            'calculation_details': result['calculation_details'],
            'formula_used': result['formula_used'], 'formula_version': d['formula_version']}
