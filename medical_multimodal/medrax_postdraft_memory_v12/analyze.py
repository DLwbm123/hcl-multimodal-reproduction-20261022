"""Audit completed post-draft interventions and distinguish repairs from draft noise."""
from collections import Counter
import json,os,re,sys
from pathlib import Path
from review_logic import ARMS,FINAL,review_text,parse_option,decision
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'medrax_hcl_v5'))
from logic import paired


def main():
    root=Path(os.environ['R981_ROOT']);read=lambda n:json.loads((root/n).read_text());finish=read('FINISHED.json');rows=read('private/episodes.json');inputs=read('private/inputs.json');states=read('private/states.json');gold=read('private/sealed_development_gold.json');requests=read('private/requests_metadata.json');events=[json.loads(x) for x in (root/'private/events.jsonl').read_text().splitlines()]
    assert finish['complete'] and finish['fatal'] is None and len(rows)==36 and all(r['status']=='completed' and r['predicted'] in list('ABCDEF') for r in rows)
    prior=root.parent/'medrax_evidence_memory_v10';prior_states=json.loads((prior/'private/states.json').read_text());assert states['experience_review']==prior_states['H_A']
    prior_inputs=json.loads((prior/'private/inputs.json').read_text());prior_results=json.loads((prior/'private/results.json').read_text());old={r['id']:r for r in prior_results['H0_static_r1']};expected_inputs=[x for x in prior_inputs if not old[x['id']]['correct']]+[x for x in prior_inputs if old[x['id']]['correct']][:3];assert inputs==expected_inputs and len(inputs)==len(gold)==6
    source=json.loads((prior/'private/source_training.json').read_text());train={r['item']['id'] for r in source};reserved={r['full_question_id'] for r in json.loads((root.parent/'medrax_learning_qualification_v9/private/members.json').read_text()) if r['role']=='confirmation_reserved'};assert len(train)==24 and len(reserved)==96 and not train&set(gold) and not reserved&(train|set(gold))
    expected=[]
    for b in [1,2]:
        for i,item in enumerate(inputs if b==1 else reversed(inputs)):
            offset=(i+b-1)%3;expected.extend((item['id'],a+'_r'+str(b)) for a in ARMS[offset:]+ARMS[:offset])
    assert [(r['id'],r['state']) for r in rows]==expected
    starts=[e for e in events if e['kind']=='controller_started'];ends=[e for e in events if e['kind']=='controller_finished'];tools=[e for e in events if e['kind']=='tool_finished'];per={}
    for e in starts:per.setdefault(e['episode'],[]).append(e)
    assert len(per)==36 and min(e['time'] for e in starts)>read('STATES_FROZEN.json')['time']
    lookup={x['id']:x for x in inputs};first_payloads={}
    for index,row in enumerate(rows):
        item=lookup[row['id']];arm=row['state'].rsplit('_r',1)[0];text=review_text(arm,states);assert row['review_context_chars']==len(text) and row['review_entered']==bool(text)
        figure=''.join(Path(p).name+' located at '+p+'\n' for p in item['images']);user='Answer this question correctly using chain of thought reasoning and carefully evaluating choices. Solve using our own vision and reasoning and thenuse tools to complement your reasoning. Trust your own judgement over any tools.\n'+item['question']+'\n'+figure
        jobs=[requests[e['key']] for e in per[index]];first=jobs[0];assert first['messages'][0]['content'][0]=={'type':'text','text':user}
        key=(row['id'],row['state'].rsplit('_r',1)[1]);first_payloads.setdefault(key,[]).append(first)
        humans=[m['content'] for m in jobs[-1]['messages'] if m['role']=='user'];assert len(humans)==(2 if arm=='H0_static' else 3) and humans[-1]==FINAL
        if text:assert humans[1]==text
        assert parse_option(row['initial_final'])==row['initial_predicted']
        if text:assert parse_option(row['review_final'])==row['review_predicted']
        assert row['memory_ids']==[r['memory_id'] for r in states[arm]['abstract']]
    assert len(first_payloads)==12 and all(len(x)==3 and all(a==x[0] for a in x) for x in first_payloads.values())
    by={label:{r['id']:r for r in rows if r['state']==label} for label in finish['metrics']};ids=list(gold);pairs=[];transitions={}
    for b in [1,2]:
        get=lambda a:list(by[a+'_r'+str(b)].values());p={k:paired(get(a),get('experience_review'),ids,gold) for k,a in [('H0_to_experience','H0_static'),('generic_to_experience','generic_review')]};assert p==finish['paired_per_repeat'][b-1];pairs.append(p)
        for a in ARMS:
            label=a+'_r'+str(b);entries=get(a);assert sum(r['predicted']==gold[r['id']] for r in entries)==finish['metrics'][label]['correct'];transitions[label]={'initial_to_review_option_changes':sum(r.get('review_entered') and r['initial_predicted']!=r.get('review_predicted') for r in entries),'initial_to_final_option_changes':sum(r['initial_predicted']!=r['predicted'] for r in entries),'review_to_final_option_changes':sum(r.get('review_entered') and r['review_predicted']!=r['predicted'] for r in entries),'initial_wrong_to_final_correct':sum(r['initial_predicted']!=gold[r['id']] and r['predicted']==gold[r['id']] for r in entries),'initial_correct_to_final_wrong':sum(r['initial_predicted']==gold[r['id']] and r['predicted']!=gold[r['id']] for r in entries)}
    g=decision(pairs,by,ids,gold,True);assert {k:v for k,v in g.items() if k!='repeat_observed_attributable_repair_ids'}==finish['acquisition_gate']
    assert len(starts)==len(ends)==read('private/call_count.json') and all(e['status']=='ok' for e in ends)
    responses=[json.loads(p.read_text()) for p in (root/'private').glob('response_*.json')];usage=Counter()
    for e in ends:
        for u in e.get('usage',[]):usage.update(u)
    result={'audit_passed':True,'physical_episodes':36,'all_final_options_valid':True,'complete':True,'error_enriched_reused_cases':6,'training_source_cases':24,'confirmation_excluded':96,'selection_reconstructed_from_v10_not_v11':True,'learned_memory_reused_unchanged':True,'initial_controller_payloads_identical_across_all_three_arms_for_each_case_and_repeat':True,'actual_post_draft_review_and_final_turns_verified':36,'all_paired_tables_and_gate_recomputed':True,'gate_passed':g['pass'],'repeat_attributable_repairs':g['repeat_observed_attributable_repair_count'],'transitions':transitions,'repeat_final_option_disagreement':{a:sum(by[a+'_r1'][k]['predicted']!=by[a+'_r2'][k]['predicted'] for k in ids) for a in ARMS},'controller_calls':len(ends),'usage':dict(usage),'completed_request_seconds':sum(r.get('seconds',0) for r in responses),'visible_429_response_count':sum('429 Too Many Requests' in r.get('stderr','') for r in responses),'tool_counts':dict(Counter(e['name'] for e in tools)),'tool_status_counts':dict(Counter(e['status'] for e in tools)),'new_memory_generation_calls':0,'original_wall_seconds':finish['seconds_since_original_start'],'final_guard':read('RUN_GUARD.json'),'medical_or_sequential_learning_established':False};(root/'AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
