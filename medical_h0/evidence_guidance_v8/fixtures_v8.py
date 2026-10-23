"""New synthetic evidence and parameter cases; never rerun the v7 panel."""
import copy
import bootstrap_v8
from fixtures import row, target


def evidence_cases():
    specs = [
        (['flag_r'],[(0,'flag_r','old',{'episode':'old_record'}),(3,'flag_r','other',{'patient':'other_record'}),(8,'flag_r','current',{})]),
        (['flag_s','flag_t'],[(2,'flag_s',41,{}),(5,'flag_t',59,{'episode':'old_record'}),(9,'flag_t',67,{})]),
        (['flag_u'],[(1,'flag_u','stable',{}),(10,'flag_u','stable',{})]),
        (['flag_v'],[(0,'flag_v','positive',{}),(11,'flag_v','negative',{})]),
        (['flag_w','flag_x'],[(1,'flag_w',31,{}),(4,'flag_x',47,{}),(8,'flag_x',53,{})]),
        (['flag_y'],[(2,'flag_y','present',{'patient':'other_record'}),(10,'flag_y','present',{'episode':'old_record'})]),
    ]
    cases=[]
    for i,(fields,entries) in enumerate(specs,1):
        patient,episode=f'synthetic_v8_e{i}',f'synthetic_visit_v8_e{i}'
        rows=[row(f'unrelated_v8_{i}_{j}',900+i*17+j,patient=patient,episode=episode) for j in range(12)]
        for index,field,value,scope in entries:
            rows[index]=row(field,value,patient=scope.get('patient',patient),episode=scope.get('episode',episode))
        prompt=('Find evidence for these exact field names: '+', '.join(fields)+'. '
            'Report all facts matching these fields in the target patient and episode, including consistent repeated measurements. '
            'Every requested field must be supported. Preserve other-patient and other-episode records as distractors, not evidence. '
            'If a requested field is absent after searching the finite input, abstain and explain what is missing. '
            'If a requested field has conflicting values, abstain and explain the conflict; there is no rule authorizing one conflicting value over another. '
            'Pages contain at most four facts. Use the available evidence and tools; do not invent facts.')
        cases.append({'id':f'E{i:02d}','stage':'navigation','required_fields':fields,'rows':rows,
                      'target':{**target(),'patient_id':patient,'episode_id':episode},'prompt':prompt})
    return cases


def mapping_cases():
    cases=[]
    for i,(dose,frequency) in enumerate([(2.5,4),(7.5,3),(15,2),(20,3)],1):
        patient,episode=f'synthetic_v8_m{i}',f'synthetic_visit_v8_m{i}'
        def r(field,value,**scope):
            return row(field,value,patient=scope.get('patient',patient),episode=scope.get('episode',episode))
        rows=[r('Unrelated measurement',101+i),r('Doses per day (doses/day)',frequency),
              r('Drug','Oxycodone'),r('Add another drug','No'),r('Dosage (mg/dose)',dose)]
        if i>=3:
            rows.insert(0,r('Doses per day (doses/day)',9,episode='old_record'))
            rows.append(r('Drug','Morphine',patient='other_record'))
        cases.append({'id':f'P{i:02d}','stage':'mapping','rows':rows,
                      'target':{**target('mme_calculator'),'patient_id':patient,'episode_id':episode}})
    patient,episode='synthetic_v8_m5','synthetic_visit_v8_m5'
    cases.append({'id':'P05','stage':'mapping','rows':[
        row('Sodium','155 mEq/L',patient=patient,episode='old_record'),
        row('Chloride','107 mEq/L',patient=patient,episode=episode),
        row('Bicarbonate','23 mEq/L',patient=patient,episode=episode),
        row('Sodium','143 mEq/L',patient=patient,episode=episode)],
        'target':{**target('anion_gap'),'patient_id':patient,'episode_id':episode}})
    patient,episode='synthetic_v8_m6','synthetic_visit_v8_m6'
    cases.append({'id':'P06','stage':'mapping','rows':[
        row('patient_description','Alert and calm (0)',patient='other_record',episode=episode),
        row('patient_description','Agitated (+2)',patient=patient,episode=episode)],
        'target':{**target('rass'),'patient_id':patient,'episode_id':episode}})
    for c in cases:
        c['prompt']=('For the synthetic patient and episode, read the facts and formula definition, bind supported parameters, '
                     f"execute {c['target']['calculator_id']}, and explicitly submit the immutable result. Use only the stated facts.")
    return cases


def schedule():
    result=[]
    for i,(e,m) in enumerate(zip(evidence_cases(),mapping_cases())):
        for c,arms in [(e,['R','I'] if i%2==0 else ['I','R']), (m,['I','R'] if i%2==0 else ['R','I'])]:
            for arm in arms:
                result.append({**copy.deepcopy(c),'case_id':c['id'],'id':c['id']+'_'+arm,'arm':arm})
    return result
