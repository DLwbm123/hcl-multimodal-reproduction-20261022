"""New synthetic mapping panel, frozen before inference."""
import copy
import bootstrap_v9
from fixtures import row,target


def cases():
    out=[]
    for i,(dose,freq) in enumerate([(3,4),(6,3),(12.5,2),(17.5,4),(8,3),(22,2)],1):
        patient,episode=f'synthetic_v9_m{i}',f'visit_v9_m{i}'
        def r(field,value,**scope):return row(field,value,patient=scope.get('patient',patient),episode=scope.get('episode',episode))
        rows=[r('Drug','Oxycodone'),r('Doses per day (doses/day)',freq),r('Dosage (mg/dose)',dose),r('Add another drug','No'),r('Unrelated measurement',701+i)]
        if i>=3:rows.insert(1,r('Doses per day (doses/day)',7,episode='old_visit'))
        if i>=5:rows.append(r('Dosage (mg/dose)',40,patient='other_patient'))
        shift=i%len(rows);rows=rows[shift:]+rows[:shift]
        out.append({'id':f'M{i:02d}','stage':'mapping','rows':rows,'target':{**target('mme_calculator'),'patient_id':patient,'episode_id':episode}})
    for i,(na,cl,hco) in enumerate([(140,102,25),(146,110,22)],7):
        patient,episode=f'synthetic_v9_m{i}',f'visit_v9_m{i}'
        rows=[row('Bicarbonate',f'{hco} mEq/L',patient=patient,episode=episode),row('Sodium',f'{na} mEq/L',patient=patient,episode=episode),row('Chloride',f'{cl} mEq/L',patient=patient,episode=episode)]
        if i==8:rows.insert(0,row('Sodium','158 mEq/L',patient=patient,episode='old_visit'))
        out.append({'id':f'M{i:02d}','stage':'mapping','rows':rows,'target':{**target('anion_gap'),'patient_id':patient,'episode_id':episode}})
    for c in out:c['prompt']=('For the synthetic patient and episode, read the facts and formula definition, bind supported parameters, '
        f"execute {c['target']['calculator_id']}, and explicitly submit the immutable result. Use only the stated facts.")
    return out


def schedule():
    return [{**copy.deepcopy(c),'case_id':c['id'],'id':c['id']+'_'+arm,'arm':arm}
            for i,c in enumerate(cases()) for arm in (['R','I'] if i%2==0 else ['I','R'])]
