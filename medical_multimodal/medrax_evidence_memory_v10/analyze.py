"""Offline final audit and attribution checks. Does not run or change any model."""
import ast
from collections import Counter
import json
import os
from pathlib import Path
import re
import sys
from memory_logic import ARMS,FIELDS,GENERIC,parse_note,render,repeated_gate
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'medrax_hcl_v5'))
from logic import paired


def main():
    root=Path(os.environ['R981_ROOT']);read=lambda n:json.loads((root/n).read_text())
    finish=read('FINISHED.json');rows=read('private/episodes.json');inputs=read('private/inputs.json');gold=read('private/sealed_development_gold.json');states=read('private/states.json');sources=read('private/source_training.json');proposals=read('private/proposals.json')
    assert finish['complete'] and finish['fatal'] is None and len(rows)==144
    assert all(r['status']=='completed' and r['predicted'] in list('ABCDEF') for r in rows)
    assert len(inputs)==len({r['case_id'] for r in inputs})==len(gold)==24
    assert all(set(r)=={'id','case_id','stage','role','question','images'} for r in inputs)
    assert {r['id'] for r in inputs}==set(gold)
    train_ids={s['item']['id'] for s in sources};assert len(sources)==len(train_ids)==24 and not train_ids&set(gold)
    reserved=[r for r in json.loads((root.parent/'medrax_learning_qualification_v9/private/members.json').read_text()) if r['role']=='confirmation_reserved']
    assert len(reserved)==96 and not {r['full_question_id'] for r in reserved}&(train_ids|set(gold))
    for i,p in enumerate(proposals):
        generation=read('private/generation_'+str(i)+'.json')
        assert not generation['tool_calls'] and generation['images']==len(sources[i]['item']['images'])
        assert generation['source_id']==p['source_id']==sources[i]['item']['id']
        assert parse_note(generation['raw_output'])=={k:v for k,v in p.items() if k!='source_id'}
    assert len(proposals)==24 and all(p['admit'] for p in proposals)
    consolidation=read('private/consolidation.json');assert not consolidation['tool_calls']
    text=consolidation['raw_output'].strip()
    if text.startswith('```'):text=re.sub(r'^```(?:json)?\s*|\s*```$','',text)
    policies=json.loads(text);assert set(policies)=={'main_policy','failure_avoidance'}
    for record in states['H_A']['abstract']:
        policy=policies[record['abstract_role']]
        assert record['note']=={k:policy[k] for k in FIELDS} and record['cue_terms']==policy['cue_terms']
        assert record['source_ids']==[proposals[i]['source_id'] for i in policy['source_indices']]
        assert set(record['source_ids'])<=train_ids
    assert len(states['H_A']['abstract'])==2
    assert states['generic_procedure']['abstract'][0]['note']==GENERIC
    expected=[]
    for block in range(2):
        for i,item in enumerate(inputs if block==0 else reversed(inputs)):
            offset=(i+block)%3
            expected.extend((item['id'],arm+'_r'+str(block+1)) for arm in ARMS[offset:]+ARMS[:offset])
    assert [(r['id'],r['state']) for r in rows]==expected
    lookup={r['id']:r for r in inputs}
    for r in rows:
        arm=r['state'].rsplit('_r',1)[0];before=json.dumps(states[arm],sort_keys=True)
        text,ids=render(states[arm],lookup[r['id']]['question'],train_ids)
        assert len(text)==r['context_chars'] and ids==r['retrieved_ids']
        assert json.dumps(states[arm],sort_keys=True)==before
    by={label:{r['id']:r for r in rows if r['state']==label} for label in finish['metrics']}
    all_ids=list(gold);strata={};gains=[];losses=[]
    for block in [1,2]:
        get=lambda arm:by[arm+'_r'+str(block)]
        a=get('H0_static');b=get('H_A');g=get('generic_procedure')
        actual={'H0_to_HA':paired(list(a.values()),list(b.values()),all_ids,gold),'generic_to_HA':paired(list(g.values()),list(b.values()),all_ids,gold),'H0_to_generic':paired(list(a.values()),list(g.values()),all_ids,gold)}
        assert actual==finish['paired_per_repeat'][block-1]
        gains.append({k for k in gold if a[k]['predicted']!=gold[k] and b[k]['predicted']==gold[k]});losses.append({k for k in gold if a[k]['predicted']==gold[k] and b[k]['predicted']!=gold[k]})
        strata['repeat_'+str(block)]={}
        for triggered in [False,True]:
            ids=[k for k in gold if bool(b[k]['retrieved_ids'])==triggered]
            strata['repeat_'+str(block)]['memory_present' if triggered else 'memory_absent']={'denominator':len(ids),'correct':{arm:sum(get(arm)[k]['predicted']==gold[k] for k in ids) for arm in ARMS},'H0_to_HA':paired(list(a.values()),list(b.values()),ids,gold),'answer_disagreement_H0_HA':sum(a[k]['predicted']!=b[k]['predicted'] for k in ids)}
    assert gains[0]==gains[1] and len(gains[0])==1 and not losses[0] and not losses[1]
    assert all(not by['H_A_r'+str(block)][k]['retrieved_ids'] and by['H_A_r'+str(block)][k]['context_chars']==0 for block in [1,2] for k in gains[0])
    jobs=read('private/gain_initial_requests.json');assert len(jobs)==4 and all(v==next(iter(jobs.values())) for v in jobs.values())
    gate=repeated_gate(finish['paired_per_repeat'],[by['H0_static_r'+str(i)] for i in [1,2]],[by['generic_procedure_r'+str(i)] for i in [1,2]],[by['H_A_r'+str(i)] for i in [1,2]],all_ids,gold,True)
    assert {k:v for k,v in gate.items() if k!='repeat_observed_acquired_ids'}==finish['acquisition_gate']
    events=[json.loads(s) for s in (root/'private/events.jsonl').read_text().splitlines()];starts=[r for r in events if r['kind']=='controller_started'];calls=[r for r in events if r['kind']=='controller_finished'];tools=[r for r in events if r['kind']=='tool_finished']
    assert len(starts)==len(calls)==470 and {r['key'] for r in starts}=={r['key'] for r in calls} and all(r['status']=='ok' for r in calls)
    assert min(r['time'] for r in starts)>read('FROZEN.json')['time']
    frozen=read('STATES_FROZEN.json')['time']
    assert max(r['time'] for r in calls if r['state'] in ['note_generation','consolidation'])<frozen<min(r['time'] for r in starts if r['phase']=='case')
    assert max(r['time'] for r in calls)<finish['time']
    usage=Counter();flags=Counter()
    for r in calls:
        for u in r.get('usage',[]):usage.update(u)
    for r in tools:
        value=ast.literal_eval(r['result'])[0];text=value.get('response','') if isinstance(value,dict) and 'response' in value else value if isinstance(value,str) else None
        if text is not None:
            if not text.strip():flags['empty']+=1
            if text.strip().lower()=='not matched':flags['not_matched']+=1
            if text.strip().lower()=='no':flags['bare_no']+=1
            if 'bibibibibibi' in text:flags['repetitive_malformed']+=1
    flips={}
    for label,entries in by.items():
        parsed=[(r,re.match(r'([A-F])(?:[).:\s]|$)',r.get('initial_final','').lstrip().lstrip('*'))) for r in entries.values()]
        flips[label]={'initial_parseable':sum(m is not None for r,m in parsed),'changed_option':sum(m is not None and m.group(1)!=r['predicted'] for r,m in parsed)}
        assert sum(r['predicted']==gold[r['id']] for r in entries.values())==finish['metrics'][label]['correct']
    result={'complete_audit_passed':True,'physical_episodes':144,'unique_development_cases':24,'training_cases':24,'all_generated_proposals_and_consolidation_reconstructed':True,'candidate_acceptance':{'accepted':24,'rejected':0,'expert_medical_validation':False},'final_policy_count':2,'policy_source_counts':[len(r['source_ids']) for r in states['H_A']['abstract']],'all_contexts_reconstructed':144,'execution_order_verified':True,'all_paired_tables_and_gate_recomputed':True,'confirmation_reserve_excluded':96,'trigger_stratified_results':strata,'same_unique_gain_case_in_both_repeats':True,'all_nominal_gain_episodes_had_zero_memory_context':True,'gain_case_initial_controller_payloads_identical_across_H0_HA_and_both_repeats':True,'memory_attributable_gain_not_established':True,'repeat_answer_disagreement':{arm:sum(by[arm+'_r1'][k]['predicted']!=by[arm+'_r2'][k]['predicted'] for k in gold) for arm in ARMS},'initial_final_consistency':flips,'controller_calls':470,'calls_by_state':dict(Counter(r['state'] for r in calls)),'controller_usage':dict(usage),'reasoning_tokens_are_subset_not_additional':True,'controller_request_seconds_sum':sum(r['seconds'] for r in calls),'generation_request_seconds':sum(r['seconds'] for r in calls if r['phase']!='case'),'controller_starts_with_images':sum(r['images']>0 for r in starts),'tool_counts':dict(Counter(r['name'] for r in tools)),'tool_statuses':dict(Counter(r['status'] for r in tools)),'tool_output_surface_flags':dict(flags),'surface_flags_not_medical_accuracy':True,'original_wall_seconds':finish['seconds_since_original_start'],'final_guard':read('RUN_GUARD.json'),'acquisition_gate_passed':False,'clinical_or_sequential_learning_claim':False}
    (root/'AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
