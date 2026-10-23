"""Frozen recovery states and disjoint confirmation cases; no model feedback."""
import copy
import bootstrap_v10
from fixtures import row,target


def make(cid,calculator,fields,outcome='answerable'):
    patient,episode='synthetic_v10_'+cid,'visit_v10_'+cid
    rows=[row(f,v,patient=scope.get('patient',patient),episode=scope.get('episode',episode)) for f,v,scope in fields]
    return {'id':cid,'case_id':cid,'stage':'mapping','rows':rows,'target':{**target(calculator),'patient_id':patient,'episode_id':episode},'expected_outcome':outcome,
            'prompt':f'For the target synthetic patient and episode, use the available facts and formula definition to bind supported parameters, execute {calculator}, and explicitly submit the immutable result. If required evidence is absent or has conflicting values without a precedence rule, abstain and explain the specific missing or conflicting field. Use only stated facts.'}


def binding(key,value,operation,*operands):
    return {'parameter':key,'value':value,'operation':operation,'operands':[{'fact_id':f'fact_{i:04d}','value':v} for i,v in operands]}


def invocation(c,bindings):return {k:v for k,v in c['target'].items() if k!='output_type'}|{'bindings':bindings}


def recovery_cases():
    cs=[]
    c=make('B01','mme_calculator',[('Drug','Oxycodone',{}),('Dosage (mg/dose)',4,{}),('Unrelated measurement',19,{}),('Doses per day (doses/day)',3,{}),('Add another drug','No',{})]);c['seed_call']=invocation(c,[binding('opioid_name','oxycodone','identity',(1,'Oxycodone')),binding('daily_dose_mg',12,'multiply',(2,4),(4,3))]);cs.append(c)
    c=make('B02','mme_calculator',[('Doses per day (doses/day)',2,{}),('Drug','Oxycodone',{}),('Dosage (mg/dose)',9,{}),('Add another drug','No',{})]);c['seed_call']=invocation(c,[binding('opioid_name','oxycodone','identity',(2,'oxycodone')),binding('daily_dose_mg',18,'multiply',(3,9),(1,2))]);cs.append(c)
    c=make('B03','anion_gap',[('Chloride','105 mEq/L',{}),('Sodium','141 mEq/L',{}),('Bicarbonate','24 mEq/L',{})]);a=binding('sodium',141,'identity',(2,141));a['unit']='mEq/L';c['seed_call']=invocation(c,[a,binding('chloride',105,'identity',(1,105)),binding('bicarbonate',24,'identity',(3,24))]);cs.append(c)
    c=make('B04','anion_gap',[('Sodium','156 mEq/L',{'episode':'old_visit'}),('Bicarbonate','26 mEq/L',{}),('Chloride','103 mEq/L',{}),('Sodium','142 mEq/L',{})]);c['seed_call']=invocation(c,[binding('sodium',156,'identity',(1,156)),binding('chloride',103,'identity',(3,103)),binding('bicarbonate',26,'identity',(2,26))]);cs.append(c)
    c=make('B05','mme_calculator',[('Drug','Oxycodone',{}),('Doses per day (doses/day)',4,{}),('Add another drug','No',{}),('Dosage (mg/dose)',5,{})]);c['seed_call']=invocation(c,[binding('opioid_name','oxycodone','lowercase',(1,'Oxycodone')),binding('daily_dose_mg',5,'multiply',(4,5),(2,4))]);cs.append(c)
    c=make('B06','anion_gap',[('Sodium','140 mEq/L',{}),('Chloride','104 mEq/L',{}),('Bicarbonate','22 mEq/L',{})]);c['seed_call']=invocation(c,[binding('sodium',142,'identity',(1,142)),binding('chloride',104,'identity',(2,104)),binding('bicarbonate',22,'identity',(3,22))]);cs.append(c)
    c=make('B07','mme_calculator',[('Drug','Oxycodone',{}),('Doses per day (doses/day)',2,{'patient':'other_patient'}),('Dosage (mg/dose)',6,{}),('Add another drug','No',{})],'missing');c['seed_call']=invocation(c,[binding('opioid_name','oxycodone','identity',(1,'Oxycodone')),binding('daily_dose_mg',6,'identity',(3,6))]);c['reason_field']='frequency';cs.append(c)
    c=make('B08','mme_calculator',[('Drug','Oxycodone',{}),('Doses per day (doses/day)',3,{}),('Dosage (mg/dose)',10,{}),('Doses per day (doses/day)',4,{}),('Add another drug','No',{})],'conflict');c['seed_call']=invocation(c,[binding('opioid_name','oxycodone','identity',(1,'Oxycodone')),binding('daily_dose_mg',30,'multiply',(3,10),(2,3))]);c['reason_field']='frequency';cs.append(c)
    return cs


def confirmation_cases():
    cs=[]
    for i,(dose,freq) in enumerate([(1.5,4),(4.5,2),(11,3),(16,2),(25,2),(30,3)],1):
        fields=[('Drug','Oxycodone',{}),('Dosage (mg/dose)',dose,{}),('Doses per day (doses/day)',freq,{}),('Add another drug','No',{})]
        if i%2==0:fields.insert(0,('Dosage (mg/dose)',8,{'patient':'other_patient'}))
        if i>=3:fields.append(('Doses per day (doses/day)',8,{'episode':'old_visit'}))
        fields.insert(i%len(fields),('Unrelated measurement',800+i,{}))
        shift=(i+1)%len(fields);fields=fields[shift:]+fields[:shift]
        cs.append(make(f'C{i:02d}','mme_calculator',fields))
    for i,(na,cl,hco) in enumerate([(139,101,23),(144,109,20),(138,100,26),(145,108,24)],7):
        fields=[('Bicarbonate',f'{hco} mEq/L',{}),('Sodium',f'{na} mEq/L',{}),('Chloride',f'{cl} mEq/L',{})]
        if i>=8:fields.insert(0,('Chloride','117 mEq/L',{'episode':'old_visit'}))
        if i>=9:fields.append(('Sodium','151 mEq/L',{'patient':'other_patient'}))
        shift=i%len(fields);fields=fields[shift:]+fields[:shift]
        cs.append(make(f'C{i:02d}','anion_gap',fields))
    c=make('C11','anion_gap',[('Sodium','143 mEq/L',{}),('Bicarbonate','25 mEq/L',{'episode':'old_visit'}),('Chloride','106 mEq/L',{})],'missing');c['reason_field']='bicarbonate';cs.append(c)
    c=make('C12','mme_calculator',[('Dosage (mg/dose)',14,{}),('Doses per day (doses/day)',2,{}),('Drug','Oxycodone',{}),('Add another drug','No',{}),('Doses per day (doses/day)',3,{})],'conflict');c['reason_field']='frequency';cs.append(c)
    return cs


def schedule(stage,candidate=None):
    cs=recovery_cases() if stage=='B' else confirmation_cases();out=[]
    for i,c in enumerate(cs):
        arms=list('ABCD');arms=arms[i%4:]+arms[:i%4] if stage=='B' else (['A',candidate] if i%2==0 else [candidate,'A'])
        for arm in arms:out.append({**copy.deepcopy(c),'id':c['id']+'_'+arm,'arm':arm,'experiment_stage':stage})
    return out
