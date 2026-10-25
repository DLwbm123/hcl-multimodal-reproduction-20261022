"""Validate real HCL restore, bank isolation, context invariance and paired gates."""
import json,os
from pathlib import Path
from task_memory import select,context,gate

def main():
    from hcl.memory import ExperienceMemory,MemoryConfig
    r=Path(os.environ['R981_ROOT']);read=lambda n:json.loads((r/n).read_text());states=read('private/states.json');items=read('private/inputs.json');b=read('budget.json');assert b['model']=='gpt-6.1-sol' and b['reasoning']=='high' and b['max_subscription_calls']==220
    memory=ExperienceMemory(MemoryConfig(record_dir=str(r/'private/cpu_memory'),raw_capacity_per_task=24,abstract_capacity_per_task=2,skill_enabled=False,anchor_selection_mode='off'))
    for arm,expected in [('H0_static',0),('H_A',24),('H_B',48)]:memory.restore(states[arm]);assert len(memory.snapshot()['raw'])==expected
    for arm in ['H0_static','H_A','H_B']:
        memory.restore(states[arm]);actual=memory.snapshot()
        for x in items:
            if x['stage']=='A' and arm=='H0_static':continue
            expected=select(states[arm],x['question'],x['stage'],arm);got=select(actual,x['question'],x['stage'],arm);assert got==expected and context(got)==context(expected)
    for x in items:
        for arm in (['H_A','H_B'] if x['stage']=='A' else ['H0_static','H_A','H_B']):
            chosen=select(states[arm],x['question'],x['stage'],arm);assert len(chosen)==(0 if arm=='H0_static' else 2)
            if chosen:assert all(v['internal_source']['task_name']=='medical_agent_'+('B' if arm=='H_B' and x['stage']=='B' else 'A') for v in chosen)
        if x['stage']=='A':assert context(select(states['H_A'],x['question'],'A','H_A'))==context(select(states['H_B'],x['question'],'A','H_B'))
    ids={'A':['a','b','c'],'B':['x','y','z','u','v','w']};gold={k:'A' for vs in ids.values() for k in vs};by={}
    for n in [1,2]:
        for label in ['B_H0_static','B_H_A','B_H_B','A_H_A','A_H_B']:
            stage=label[0];by[label+'_r'+str(n)]={k:{'predicted':'A' if label in ['B_H_B','A_H_A','A_H_B'] else 'B','retrieved_ids':['B0','B1'] if label=='B_H_B' else [],'context_text':'B' if label=='B_H_B' else 'A'} for k in ids[stage]}
    pairs=[{'B_H0_to_HB':{'gain':2,'loss':0},'B_HA_to_HB':{'gain':1,'loss':0},'A_HA_to_HB':{'gain':0,'loss':0}} for _ in [1,2]];assert gate(pairs,by,ids,gold,True)['pass'] and not gate(pairs,by,ids,gold,False)['pass'];by['B_H_B_r1']={k:{**v,'context_text':'A'} for k,v in by['B_H_B_r1'].items()};assert not gate(pairs,by,ids,gold,True)['pass']
    v={'passed':True,'real_HCL_restore_keeps24A_and48AB_records':True,'question_only_top2_task_bank_isolation_checked':True,'A_HA_HB_actual_context_identical':True,'no_generation_or_historical_image_inference':True,'actual_B_context_update_and_complete_retention_gate_checked':True,'real_model_calls':0};(r/'CPU_CHECK.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))

if __name__=='__main__':main()
