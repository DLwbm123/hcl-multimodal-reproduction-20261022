"""Evaluator-only extraction from visible facts. Never imported by model tools.

No benchmark answers or case identifiers are used here. These rules establish
reference parameters and admissible evidence; they do not repair model calls.
"""
import math
import re
import unicodedata


class Unresolved(ValueError):
    pass


def norm(s):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(s)).casefold()).strip()


class Evidence:
    def __init__(self, rows, patient, episode):
        self.rows = rows
        self.patient = patient
        self.episode = episode
        self.bindings = {}

    def field(self, *names, stable=False):
        names = {norm(n) for n in names}
        matches = [(f'fact_{i+1:04d}', r) for i, r in enumerate(self.rows)
                   if r['patient_id'] == self.patient and norm(r['field_name']) in names
                   and (stable or r['episode_id'] == self.episode)]
        if not matches:
            raise Unresolved('Missing visible field: ' + ', '.join(sorted(names)))
        values = {norm(r['value_text']) for _, r in matches}
        if len(values) != 1:
            raise Unresolved('Ambiguous visible field: ' + ', '.join(sorted(names)))
        return matches[0][1]['value_text'], [fid for fid, _ in matches]

    def numeric(self, *names, pattern=None, stable=False):
        text, ids = self.field(*names, stable=stable)
        match = re.search(pattern, text, re.I) if pattern else re.match(r'^\s*([+-]?\d+(?:\.\d+)?)\b', text)
        if not match or (not pattern and re.match(r'\s*[-–]\s*\d', text[match.end():])):
            raise Unresolved('Not an explicit scalar measurement: ' + names[0])
        value = float(match[1])
        if not math.isfinite(value):
            raise Unresolved('Nonfinite measurement')
        return value, [ids]

    def boolean(self, *names):
        text, ids = self.field(*names)
        if re.match(r'^yes\b', norm(text)):
            return True, [ids]
        if re.match(r'^no\b', norm(text)):
            return False, [ids]
        raise Unresolved('No explicit yes/no: ' + names[0])

    def category(self, names, mapping, stable=False):
        text, ids = self.field(*names, stable=stable)
        key = norm(text.split('(')[0])
        if key not in mapping:
            raise Unresolved('Unreviewed category: ' + names[0])
        return mapping[key], [ids]

    def put(self, key, pair, derivation='direct', integral=False):
        value, groups = pair
        if integral:
            if value != int(value):
                raise Unresolved('Nonintegral value for ' + key)
            value = int(value)
        self.bindings[key] = {'value': value, 'required_source_groups': groups,
                              'derivation': derivation}
        return value

    def gcs(self):
        specs = [(['Best eye response', 'best_eye_response'],
                  {'to verbal command': 3, 'spontaneous': 4, 'to pain': 2, 'none': 1}),
                 (['Best verbal response', 'best_verbal_response'],
                  {'inappropriate words': 3, 'confused': 4, 'oriented': 5, 'none': 1}),
                 (['Best motor response', 'best_motor_response'],
                  {'withdrawal from pain': 4, 'obeys commands': 6, 'localizes pain': 5, 'none': 1})]
        parts = [self.category(names, table) for names, table in specs]
        return sum(x[0] for x in parts), sum((x[1] for x in parts), [])


def reference_bindings(cid, rows, patient, episode):
    e = Evidence(rows, patient, episode)
    if cid == 'anion_gap':
        for p, f in [('sodium', 'Sodium'), ('chloride', 'Chloride'), ('bicarbonate', 'Bicarbonate')]:
            e.put(p, e.numeric(f))
        # Algorithm choice belongs to the declared uncorrected/no-K output
        # contract, not to a fabricated patient observation. Optional library
        # arguments are omitted, never given invented clinical values.
    elif cid == 'winters_formula':
        e.put('hco3', e.numeric('Bicarbonate'))
    elif cid == 'mme_calculator':
        e.put('opioid_name', e.category(['Drug'], {'oxycodone': 'oxycodone'}), 'category')
        dose, ds = e.numeric('Dosage (mg/dose)')
        freq, fs = e.numeric('Doses per day (doses/day)')
        e.put('daily_dose_mg', (dose * freq, ds + fs), 'dose_times_frequency')
        extra, _ = e.field('Add another drug')
        if norm(extra) != 'no':
            raise Unresolved('Multi-drug MME not in this subdomain')
    elif cid == 'ich_score':
        e.put('gcs_score', e.numeric('Glasgow Coma Score category', pattern=r'GCS\s*=\s*(\d+)'), integral=True)
        length, a = e.numeric('Hemorrhage Length (A)')
        width, b = e.numeric('Hemorrhage Width (B)')
        slices, c = e.numeric('Number of CT Slices with hemorrhage (C component)')
        thickness, d = e.numeric('CT Slice Thickness')
        shape, shape_ids = e.field('Hemorrhage Shape')
        if norm(shape) != 'round or ellipsoid':
            raise Unresolved('ABC/2 shape assumption not explicit')
        e.put('ich_volume_ml', (length * width * slices * thickness / 10 / 2, a+b+c+d+[shape_ids]), 'abc_over_two_mm_to_cm')
        e.put('ivh_present', e.boolean('Intraventricular hemorrhage'))
        e.put('infratentorial', e.boolean('Infratentorial origin of hemorrhage'))
        e.put('age', e.numeric('Age ≥80 years', pattern=r'\((\d+)\s*years'), integral=True)
    elif cid in ('qsofa_score', 'sofa_score'):
        if cid == 'qsofa_score':
            e.put('respiratory_rate', e.numeric('respiratory_rate_≥22_per_min', 'Respiratory rate ≥22', pattern=r'\((\d+)\)'), integral=True)
            e.put('systolic_bp', e.numeric('systolic_bp_≤100_mmHg', 'Systolic BP ≤100', pattern=r'\((\d+)\)'), integral=True)
            gcs, sources = e.gcs()
            e.put('gcs_score', (gcs, sources), 'sum_gcs_components', integral=True)
            e.put('altered_mentation', (gcs < 15, sources), 'gcs_below_fifteen')
        else:
            pao2, ps = e.numeric('PaO2 mm Hg', 'PaO2_mmHg')
            fio2, fs = e.numeric('FiO2 %', 'FiO2_percent')
            e.put('pao2_fio2_ratio', (pao2 / (fio2 / 100), ps + fs), 'pao2_over_fio2_fraction')
            for p, names in [('platelets', ['Platelets, ×10³/µL', 'platelets_x10^3_per_uL']),
                             ('bilirubin', ['Bilirubin, mg/dL', 'bilirubin_mg_per_dL']),
                             ('creatinine', ['Creatinine, mg/dL', 'creatinine_mg_per_dL']),
                             ('map_value', ['Mean arterial pressure (MAP) mmHg', 'mean_arterial_pressure_mmHg'])]:
                e.put(p, e.numeric(*names))
            try:
                gc = e.numeric('Glasgow Coma Scale')
            except Unresolved:
                gc = e.gcs()
            e.put('gcs_score', gc, 'gcs_value_or_component_sum', integral=True)
            value, ids = e.field('Vasoactive agents', 'vasoactive_agents_administered')
            if norm(value) != 'none':
                raise Unresolved('Dose mapping for active vasopressors not reviewed')
            for key in ['dopamine_dose', 'epinephrine_dose', 'norepinephrine_dose']:
                e.put(key, (0.0, [ids]), 'explicit_no_vasoactive_agents')
            e.put('dobutamine_any', (False, [ids]), 'explicit_no_vasoactive_agents')
            e.put('is_mechanically_ventilated', e.boolean('On mechanical ventilation (Including CPAP)', 'on_mechanical_ventilation_including_CPAP'))
    elif cid == 'rass':
        e.put('rass_score', e.category(['patient_description'], {'restless': 1, 'alert and calm': 0, 'agitated': 2}), 'category')
    elif cid == 'iss':
        fields = {'head_neck_ais': 'Head and neck worst injury', 'face_ais': 'Face worst injury',
                  'chest_ais': 'Chest worst injury', 'abdomen_ais': 'Abdomen worst injury',
                  'extremity_ais': 'Extremity (including pelvis) worst injury', 'external_ais': 'External worst injury'}
        for p, f in fields.items():
            e.put(p, e.category([f], {'none': 0, 'minor': 1, 'moderate': 2, 'serious': 3,
                                     'severe': 4, 'critical': 5, 'unsurvivable': 6}), 'ais_category')
    elif cid == 'perc_rule':
        fields = {'age_50_or_older': 'Age ≥50', 'heart_rate_100_or_higher': 'HR ≥100',
                  'o2_sat_below_95': 'O2 sat on room air <95%', 'unilateral_leg_swelling': 'Unilateral leg swelling',
                  'hemoptysis': 'Hemoptysis', 'recent_surgery_trauma': 'Recent surgery or trauma',
                  'prior_pe_dvt': 'Prior PE or DVT', 'hormone_use': 'Hormone use'}
        for p, f in fields.items():
            e.put(p, e.boolean(f))
    elif cid == 'spesi':
        for p, f in [('age', 'Age, years'), ('heart_rate', 'Heart rate, bpm'), ('systolic_bp', 'Systolic BP, mmHg')]:
            e.put(p, e.numeric(f), integral=True)
        e.put('spo2', e.numeric('O2 saturation'))
        e.put('cancer', e.boolean('History of cancer'))
        e.put('chronic_cardiopulmonary_disease', e.boolean('History of chronic cardiopulmonary disease'))
    elif cid == 'ideal_body_weight':
        e.put('height_cm', e.numeric('Height'))
        e.put('sex', e.category(['Sex'], {'male': 'male', 'female': 'female'}), 'category')
    elif cid == 'corrected_qt':
        e.put('qt_interval', e.numeric('Measured QT interval'))
        e.put('heart_rate', e.numeric('Heart rate'))
        e.put('formula', e.category(['Correction formula'], {'bazett': 'bazett', 'fridericia': 'fridericia', 'framingham': 'framingham'}), 'category')
        e.put('sex', e.category(['Sex'], {'male': 'male', 'female': 'female'}, stable=True), 'same_patient_stable_attribute')
    elif cid == 'ckd_epi_2021':
        eq, _ = e.field('Equation')
        if norm(eq) != '2021 ckd-epi creatinine':
            raise Unresolved('Equation version is not explicit')
        e.put('age', e.numeric('Age'), integral=True)
        e.put('sex', e.category(['Sex'], {'male': 'male', 'female': 'female'}), 'category')
        e.put('serum_creatinine', e.numeric('Serum creatinine'))
    elif cid == 'mallampati_score':
        e.put('mallampati_class', e.category(['oropharyngeal_visualization'], {'soft palate not visualized': 4,
              'hard palate only visible': 4}), 'category')
    elif cid == 'asa_physical_status':
        value, ids = e.field('ASA Class')
        roman = norm(value).split(' ')[0]
        mapping = {'i': 1, 'ii': 2, 'iii': 3, 'iv': 4, 'v': 5, 'vi': 6}
        if roman not in mapping:
            raise Unresolved('ASA class not explicit')
        e.put('asa_class', (mapping[roman], [ids]), 'roman_class')
        e.put('is_emergency', e.boolean('Emergency surgery'))
    elif cid == 'rcri':
        fields = {'high_risk_surgery': 'High-risk surgery (intraperitoneal, intrathoracic, suprainguinal vascular)',
                  'ischemic_heart_disease': 'History of ischemic heart disease', 'heart_failure': 'History of congestive heart failure',
                  'cerebrovascular_disease': 'History of cerebrovascular disease (TIA/stroke)',
                  'insulin_diabetes': 'Pre-operative treatment with insulin',
                  'creatinine_above_2': 'Pre-operative creatinine >2 mg/dL (176.8 µmol/L)'}
        for p, f in fields.items():
            e.put(p, e.boolean(f))
    else:
        raise Unresolved('No bounded-subdomain rule for ' + cid)
    return e.bindings
