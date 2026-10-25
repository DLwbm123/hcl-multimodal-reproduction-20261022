"""Audit executed visual crops, original-image retention, source isolation and all paired results."""
from collections import Counter
import json,os,sys
from pathlib import Path
from focus_logic import ARMS,FINAL,FIXED,CATALOG,decision
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'medrax_hcl_v5'))
from logic import paired

def main():
    root=Path(os.environ['R981_ROOT']);read=lambda n:json.loads((root/n).read_text());finish=read('FINISHED.json');rows=read('private/episodes.json');inputs=read('private/inputs.json');states=read('private/states.json');gold=read('private/sealed_development_gold.json');requests=read('private/requests_metadata.json');events=[json.loads(x) for x in (root/'private/events.jsonl').read_text().splitlines()]
    assert finish['complete'] and finish['fatal'] is None and len(rows)==36 and all(r['status']=='completed' and r['predicted'] in list('ABCDEF') for r in rows)
    assert states['H0_static']['policy'] is None and states['fixed_focus']['policy']==dict(crop_ids=FIXED)
    source=read('private/source_training.json');assert len(source)==24
    prior=root.parent/'medrax_learning_qualification_v9';v7=root.parent/'medrax_corrective_memory_v7';load=lambda r,n:json.loads((r/n).read_text());train={r['id'] for r in load(prior,'private/inputs.json') if r['stage']=='B' and r['role']=='train'};reserved={r['full_question_id'] for r in load(prior,'private/members.json') if r['role']=='confirmation_reserved'}
    assert len(train)==24 and len(reserved)==96 and not train&set(gold) and not reserved&(train|set(gold))
    pool=[x for x in load(v7,'private/inputs.json') if x['stage']=='B' and x['role']=='report'];old={r['id']:r for r in load(v7,'private/report_results.json')['H0_static'] if r['stage']=='B'};assert [x['id'] for x in inputs]==[x['id'] for x in pool if not old[x['id']]['correct']][:3]+[x['id'] for x in pool if old[x['id']]['correct']][:3]
    expected=[]
    for b in [1,2]:
        for i,item in enumerate(inputs if b==1 else reversed(inputs)):
            offset=(i+b-1)%3;expected.extend((item['id'],a+'_r'+str(b)) for a in ARMS[offset:]+ARMS[:offset])
    assert [(r['id'],r['state']) for r in rows]==expected
    starts=[e for e in events if e['kind']=='controller_started'];ends=[e for e in events if e['kind']=='controller_finished'];tools=[e for e in events if e['kind']=='tool_finished'];per={}
    for e in starts:per.setdefault(e['episode'],[]).append(e)
    assert len(per[None])==1 and requests[per[None][0]['key']]['image_count']==0
    assert min(e['time'] for e in starts if e['episode'] is not None)>read('STATES_FROZEN.json')['time']
    lookup={x['id']:x for x in inputs};crop_total=0;different_cases=0
    for index,row in enumerate(rows):
        item=lookup[row['id']];arm=row['state'].rsplit('_r',1)[0];views=read('private/views_'+str(index)+'.json');policy=states[arm]['policy'];assert row['crop_images']==len(views)==(0 if policy is None else 4*len(item['images']))
        expected_paths=list(item['images']);crop_total+=len(views);different_cases+=bool(row['visual_input_differs_from_fixed'])
        for i,v in enumerate(views):
            assert v['current_image_index']==i//4+1 and v['normalized_box']==CATALOG[policy['crop_ids'][i%4]] and v['output_size']==[512,512];expected_paths.append(v['view_path'])
        jobs=[requests[e['key']] for e in per[index]];first=jobs[0];assert first['image_count']==row['controller_image_count']==len(expected_paths)
        actual_paths=[part['path'] for part in first['messages'][0]['content'] if part['type']=='image'];assert actual_paths==expected_paths
        text=first['messages'][0]['content'][0]['text'];figure=''.join(Path(p).name+' located at '+p+'\n' for p in item['images']);native='Answer this question correctly using chain of thought reasoning and carefully evaluating choices. Solve using our own vision and reasoning and thenuse tools to complement your reasoning. Trust your own judgement over any tools.\n'+item['question']+'\n'+figure
        assert text.startswith(native) and len(text)-len(native)==row['visual_context_chars']
        if policy is None:assert text==native
        else:assert text.endswith(json.dumps(views,ensure_ascii=False))
        humans=[m['content'] for m in jobs[-1]['messages'] if m['role']=='user'];assert len(humans)==2 and humans[-1]==FINAL
    by={label:{r['id']:r for r in rows if r['state']==label} for label in finish['metrics']};ids=list(gold);pairs=[]
    for b in [1,2]:
        get=lambda a:list(by[a+'_r'+str(b)].values());pairs.append({k:paired(get(a),get('learned_focus'),ids,gold) for k,a in [('H0_to_learned','H0_static'),('fixed_to_learned','fixed_focus')]})
        for a in ARMS:assert sum(r['predicted']==gold[r['id']] for r in get(a))==finish['metrics'][a+'_r'+str(b)]['correct']
    assert pairs==finish['paired_per_repeat'];g=decision(pairs,by,ids,gold,True);assert {k:v for k,v in g.items() if k!='repeated_gain_ids'}==finish['acquisition_gate']
    assert len(starts)==len(ends)==read('private/call_count.json') and all(e['status']=='ok' for e in ends)
    responses=[json.loads(p.read_text()) for p in (root/'private').glob('response_*.json')];assert len(responses)==len(ends);usage=Counter()
    for e in ends:
        for u in e.get('usage',[]):usage.update(u)
    result={'audit_passed':True,'physical_episodes':36,'source_cases':24,'confirmation_excluded':96,'source_only_single_proposal_no_images':True,'original_selection_and_execution_order_verified':True,'actual_original_and_crop_paths_in_controller_payloads_verified':36,'original_two_turn_final_scoring_verified':36,'policy_frozen_before_targets':True,'real_crop_views_in_payloads':crop_total,'learned_case_repeats_reporting_actual_pixel_difference':different_cases,'remote_pixel_check':read('REMOTE_PIXEL_CHECK.json'),'final_option_disagreements_between_arms':sum(len({by[a+'_r'+str(b)][k]['predicted'] for a in ARMS})>1 for b in [1,2] for k in ids),'repeat_final_option_disagreement':{a:sum(by[a+'_r1'][k]['predicted']!=by[a+'_r2'][k]['predicted'] for k in ids) for a in ARMS},'gate_passed':g['pass'],'repeated_gains':g['repeated_gain_with_actual_different_pixels_count'],'all_paired_tables_and_gate_recomputed':True,'controller_calls_including_one_proposal':len(ends),'usage':dict(usage),'completed_request_seconds':sum(r.get('seconds',0) for r in responses),'visible_429_response_count':sum('429 Too Many Requests' in r.get('stderr','') for r in responses),'tool_counts':dict(Counter(e['name'] for e in tools)),'tool_status_counts':dict(Counter(e['status'] for e in tools)),'original_wall_seconds':finish['seconds_since_original_start'],'native_tool_load_seconds_after_program_freeze':read('TOOLS_LOADED.json')['time']-read('STATES_FROZEN.json')['time'],'final_guard':read('RUN_GUARD.json'),'medical_or_sequential_learning_established':False};(root/'AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
