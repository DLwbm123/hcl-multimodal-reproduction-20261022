"""Learning-only, human-reviewed input mapping. Never imported by model tools.

Reads visible facts only; no target answers, expected numbers or evaluator scores.
Unresolvable raw measurements remain blocked rather than being invented.
"""
import re
from clinical_catalog import norm

class UnresolvedInput(ValueError):pass

def mapped_inputs(cid,rows,episode):
    facts=[r for r in rows if r['episode_id']==episode];used=[]
    def get(*names):
        matches=[r for r in facts if norm(r['field_name']) in {norm(n) for n in names}]
        if len(matches)!=1:raise UnresolvedInput('missing or ambiguous visible field: '+names[0])
        r=matches[0];used.append({'field':r['field_name'],'value':r['value_text'],'episode_id':r['episode_id'],'unit':r['unit']});return r['value_text']
    def yes(*names):
        s=get(*names).lower()
        if s.startswith('yes'):return True
        if s.startswith('no'):return False
        raise UnresolvedInput('not an explicit yes/no: '+names[0])
    def num(*names):
        s=get(*names)
        m=re.match(r'^\s*([+-]?\d+(?:\.\d+)?)\b',s)
        if not m:raise UnresolvedInput('no exact leading measurement: '+names[0])
        tail=s[m.end():]
        if re.match(r'\s*[-–]\s*\d',tail):raise UnresolvedInput('interval measurement requires range-capable formula: '+names[0])
        return float(m[1])
    def category(names,mapping):
        v=get(*names).lower().split('(')[0].strip()
        if v not in mapping:raise UnresolvedInput('unmapped category: '+names[0])
        return mapping[v]
    def gcs():
        return {'eye_response':category(['Best eye response','best_eye_response'],{'to verbal command':3,'spontaneous':4,'to pain':2,'none':1}),
                'verbal_response':category(['Best verbal response','best_verbal_response'],{'inappropriate words':3,'confused':4,'oriented':5,'none':1}),
                'motor_response':category(['Best motor response','best_motor_response'],{'withdrawal from pain':4,'obeys commands':6,'localizes pain':5,'none':1})}
    if cid=='stop_bang':
        p=dict(snoring=yes('snoring_loudly'),tired=yes('daytime_tiredness'),observed_apnea=yes('observed_apneas'),high_blood_pressure=yes('hypertension_treated'),bmi_over_35=num('BMI')>35,age_over_50=num('age')>50,neck_over_40cm=num('neck_circumference')>40,male_gender=category(['gender'],{'male':True,'female':False}))
    elif cid=='rass':p={'rass_score':category(['patient_description'],{'restless':1,'alert and calm':0,'agitated':2})}
    elif cid=='glasgow_coma_scale':
        p=gcs();p['is_intubated']=False # Reference-only interpretation of an explicitly assessable verbal response; retain in provenance audit.
    elif cid=='shock_index':
        # Demographic type affects interpretation, not raw HR/SBP. No patient type is assumed.
        raise UnresolvedInput('patient_type not established in this episode; raw HR/SBP possible but full library interpretation requires type')
    elif cid=='iss':
        fields={'head_neck_ais':'Head and neck worst injury','face_ais':'Face worst injury','chest_ais':'Chest worst injury','abdomen_ais':'Abdomen worst injury','extremity_ais':'Extremity (including pelvis) worst injury','external_ais':'External worst injury'}
        p={k:category([v],{'minor':1,'moderate':2,'serious':3,'severe':4,'critical':5,'unsurvivable':6,'none':0}) for k,v in fields.items()}
    elif cid=='mascc_score':
        p={'burden_of_illness':category(['burden_of_illness'],{'none or mild':'none_mild','moderate':'moderate','severe':'severe'}),'no_hypotension':not yes('hypotension_sbp_lt_90_mm_hg'),'no_copd':not yes('active_copd'),'solid_tumor_or_no_fungal_hx':category(['type_of_cancer'],{'solid tumor':True}),'no_dehydration':not yes('dehydration_requiring_iv_fluids'),'outpatient_status':category(['status_at_onset_of_fever'],{'outpatient':True,'inpatient':False}),'age_lt_60':num('age_years')<60}
    elif cid=='news2_score':
        oxygen=get('Room air or supplemental O₂','oxygen_supplementation').lower()
        if not (oxygen.startswith('supplemental') or oxygen.startswith('yes') or oxygen.startswith('no') or oxygen=='room air'):raise UnresolvedInput('unknown oxygen state')
        p={'respiratory_rate':int(num('Respiratory rate, breaths per minute','respiratory_rate_breaths_per_min')),'spo2':int(num('SpO₂')),'on_supplemental_o2':oxygen.startswith(('supplemental','yes')),'temperature':num('Temperature','temperature_C'),'systolic_bp':int(num('Systolic BP, mmHg','systolic_bp_mmHg')),'heart_rate':int(num('Pulse, beats per minute','pulse_beats_per_min')),'consciousness':category(['Consciousness'],{'alert':'A','new-onset confusion/responds to voice':'C'}),'use_scale_2':yes('Hypercapnic respiratory failure','hypercapnic_respiratory_failure')}
    elif cid=='psi_port':
        fields={'nursing_home_resident':'Nursing home resident','neoplastic_disease':'Neoplastic disease','liver_disease':'Liver disease history','chf':'CHF history','cerebrovascular_disease':'Cerebrovascular disease history','renal_disease':'Renal disease history','altered_mental_status':'Altered mental status','respiratory_rate_gte_30':'Respiratory rate ≥30 breaths/min','systolic_bp_lt_90':'Systolic blood pressure <90 mmHg','temperature_abnormal':'Temperature <35°C (95°F) or >39.9°C (103.8°F)','pulse_gte_125':'Pulse ≥125 beats/min','arterial_ph_lt_7_35':'pH <7.35','bun_gte_30':'BUN ≥30 mg/dL or ≥11 mmol/L','sodium_lt_130':'Sodium <130 mmol/L','glucose_gte_250':'Glucose ≥250 mg/dL or ≥14 mmol/L','hematocrit_lt_30':'Hematocrit <30%','pao2_lt_60_or_sao2_lt_90':'Partial pressure of oxygen <60 mmHg or <8 kPa','pleural_effusion':'Pleural effusion on x-ray'}
        p={k:yes(v) for k,v in fields.items()};p.update(age_years=int(num('Age')),female=category(['Sex'],{'female':True,'male':False}))
    elif cid=='qsofa_score':
        rr=get('Respiratory rate ≥22','respiratory_rate_≥22_per_min');sbp=get('Systolic BP ≤100','systolic_bp_≤100_mmHg')
        def parenthetic_measure(s):
            m=re.search(r'\((\d+)\)',s)
            if not m:raise UnresolvedInput('threshold-only fact cannot be inverted to raw vital sign')
            return int(m[1])
        q=gcs();p={'respiratory_rate':parenthetic_measure(rr),'systolic_bp':parenthetic_measure(sbp),'gcs_score':sum(q.values()),'altered_mentation':sum(q.values())<15}
    elif cid=='sofa_score':
        try:g=int(num('Glasgow Coma Scale'))
        except UnresolvedInput:g=sum(gcs().values())
        vaso=get('Vasoactive agents','vasoactive_agents_administered')
        if vaso.lower()!='none':raise UnresolvedInput('vasopressor dose mapping not reviewed')
        p={'pao2_fio2_ratio':num('PaO2 mm Hg','PaO2_mmHg')/(num('FiO2 %','FiO2_percent')/100),'platelets':num('Platelets, ×10³/µL','platelets_x10^3_per_uL'),'bilirubin':num('Bilirubin, mg/dL','bilirubin_mg_per_dL'),'gcs_score':g,'creatinine':num('Creatinine, mg/dL','creatinine_mg_per_dL'),'map_value':num('Mean arterial pressure (MAP) mmHg','mean_arterial_pressure_mmHg'),'dopamine_dose':0,'epinephrine_dose':0,'norepinephrine_dose':0,'dobutamine_any':False,'is_mechanically_ventilated':yes('On mechanical ventilation (Including CPAP)','on_mechanical_ventilation_including_CPAP')}
    elif cid=='wells_pe':
        fields={'clinical_signs_dvt':'Clinical signs and symptoms of DVT','pe_most_likely_diagnosis':'PE is #1 diagnosis OR equally likely','heart_rate_gt_100':'Heart rate >100','immobilization_or_surgery':'Immobilization at least 3 days OR surgery in the previous 4 weeks','previous_dvt_pe':'Previous, objectively diagnosed PE or DVT','hemoptysis':'Hemoptysis','malignancy':'Malignancy w/ treatment within 6 months or palliative'}
        p={k:yes(v) for k,v in fields.items()}
    elif cid=='perc_rule':
        fields={'age_50_or_older':'Age ≥50','heart_rate_100_or_higher':'HR ≥100','o2_sat_below_95':'O2 sat on room air <95%','unilateral_leg_swelling':'Unilateral leg swelling','hemoptysis':'Hemoptysis','recent_surgery_trauma':'Recent surgery or trauma','prior_pe_dvt':'Prior PE or DVT','hormone_use':'Hormone use'};p={k:yes(v) for k,v in fields.items()}
    elif cid=='spesi':p={'age':int(num('Age, years')),'cancer':yes('History of cancer'),'chronic_cardiopulmonary_disease':yes('History of chronic cardiopulmonary disease'),'heart_rate':int(num('Heart rate, bpm')),'systolic_bp':int(num('Systolic BP, mmHg')),'spo2':num('O2 saturation')}
    else:raise UnresolvedInput('no independently reviewed field mapping')
    return p,used
