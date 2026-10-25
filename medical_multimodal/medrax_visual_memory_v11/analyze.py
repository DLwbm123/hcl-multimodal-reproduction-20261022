"""Audit the stopped run without further inference; keep partial rounds explicit."""
from collections import Counter
import json,os,re,sys
from pathlib import Path
from episodic import ARMS,choose,context,gate
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'medrax_hcl_v5'))
from logic import paired


def main():
    root=Path(os.environ['R981_ROOT']);read=lambda n:json.loads((root/n).read_text())
    finish=read('FINISHED.json');rows=read('private/episodes.json');inputs=read('private/inputs.json');states=read('private/states.json');support=read('private/support_selection.json');gold=read('private/sealed_development_gold.json');requests=read('private/requests_metadata.json')
    events=[json.loads(x) for x in (root/'private/events.jsonl').read_text().splitlines()]
    assert len(inputs)==len(gold)==24 and {x['id'] for x in inputs}==set(gold)
    assert all(set(x)=={'id','case_id','stage','role','question','images'} for x in inputs)
    records=states['historical_memory']['raw'];assert len(records)==24
    train_ids={r['internal_source']['task_id'] for r in records};assert len(train_ids)==24 and not train_ids&set(gold)
    reserve=json.loads((root.parent/'medrax_learning_qualification_v9/private/members.json').read_text());reserved={r['full_question_id'] for r in reserve if r['role']=='confirmation_reserved'}
    assert len(reserved)==96 and not reserved&(train_ids|set(gold))
    expected=[]
    for b in [1,2]:
        for i,item in enumerate(inputs if b==1 else reversed(inputs)):
            offset=(i+b-1)%4;expected.extend((item['id'],a+'_r'+str(b)) for a in ARMS[offset:]+ARMS[:offset])
    assert [(r['id'],r['state']) for r in rows]==expected[:len(rows)]
    assert len(rows)==finish['physical_episodes'] and len(rows)<=192
    valid=lambda r:r.get('status')=='completed' and r.get('predicted') in list('ABCDEF')
    starts=[e for e in events if e['kind']=='controller_started'];ends=[e for e in events if e['kind']=='controller_finished'];tools=[e for e in events if e['kind']=='tool_finished'];first={}
    for e in starts:first.setdefault(e['episode'],e)
    assert len(first)==len(rows) and min(e['time'] for e in starts)>read('STATES_FROZEN.json')['time']
    lookup={r['id']:r for r in inputs}
    for index,row in enumerate(rows):
        item=lookup[row['id']];arm=row['state'].rsplit('_r',1)[0];text='';extra=[];retrieved=[];history_id=None
        if arm!='H0_static':
            rec,wrong=choose(records,item['question']);assert support[item['id']]=={'selected':rec['memory_id'],'mismatch':wrong['memory_id']}
            visual=arm in ['history_image','history_mismatch'];text=context(rec,visual);retrieved=[rec['memory_id']]
            if visual:
                chosen=rec if arm=='history_image' else wrong;extra=chosen['images'];history_id=chosen['memory_id']
        assert row['context_chars']==len(text) and row['retrieved_ids']==retrieved and row['history_image_count']==len(extra) and row['history_image_memory_id']==history_id
        paths=item['images'];figure=''.join(Path(p).name+' located at '+p+'\n' for p in paths)
        user='Answer this question correctly using chain of thought reasoning and carefully evaluating choices. Solve using our own vision and reasoning and thenuse tools to complement your reasoning. Trust your own judgement over any tools.\n'+item['question']+'\n'+figure+text
        job=requests[first[index]['key']];content=job['messages'][0]['content']
        assert content[0]=={'type':'text','text':user}
        assert [p['path'] for p in content if p.get('type')=='image']==paths+extra
        assert job['image_count']==len(paths+extra)
    for e in starts:
        arm=e['state'].rsplit('_r',1)[0];assert e['images']==(2 if arm in ['history_image','history_mismatch'] else 1)
    by={label:{r['id']:r for r in rows if r['state']==label} for label in finish['metrics']};ids=list(gold);pairs=[];observed={};full_blocks=[]
    for b in [1,2]:
        get=lambda arm:by[arm+'_r'+str(b)]
        for arm in ARMS:
            actual=list(get(arm).values());metric=finish['metrics'][arm+'_r'+str(b)]
            assert metric['valid']==sum(valid(r) for r in actual) and metric['correct']==sum(r.get('predicted')==gold[r['id']] for r in actual)
        p={key:paired(list(get(arm).values()),list(get('history_image').values()),ids,gold) for key,arm in [('H0_to_image','H0_static'),('text_to_image','history_text'),('mismatch_to_image','history_mismatch')]};assert p==finish['paired_per_repeat'][b-1];pairs.append(p)
        common=[k for k in ids if all(k in get(a) and valid(get(a)[k]) for a in ARMS)]
        observed[str(b)]={'all_four_completed_cases':len(common),'correct_on_same_completed_cases':{a:sum(get(a)[k]['predicted']==gold[k] for k in common) for a in ARMS},'aligned_mismatch_answer_disagreement':sum(get('history_image')[k]['predicted']!=get('history_mismatch')[k]['predicted'] for k in common)}
        if len(common)==24:full_blocks.append(b)
    complete=finish['fatal'] is None and len(rows)==192 and all(valid(r) for r in rows);decision=gate(pairs,by,ids,gold,complete);assert {k:v for k,v in decision.items() if k!='repeat_observed_acquired_ids'}==finish['acquisition_gate']
    usage=Counter()
    for e in ends:
        for u in e.get('usage',[]):usage.update(u)
    responses=[json.loads(p.read_text()) for p in (root/'private').glob('response_*.json')];waits=[]
    for r in responses:waits.extend(map(float,re.findall(r'request \(\d+/\d+ in ([0-9.]+)s\)',r.get('stderr',''))))
    flips={}
    for label,entries in by.items():
        parsed=[(r,re.match(r'([A-F])(?:[).:\s]|$)',r.get('initial_final','').lstrip().lstrip('*'))) for r in entries.values() if valid(r)]
        flips[label]={'parseable_initial':sum(m is not None for r,m in parsed),'initial_final_option_changed':sum(m is not None and m.group(1)!=r['predicted'] for r,m in parsed)}
    result={'audit_passed':True,'planned_episodes':192,'physical_episodes':len(rows),'completed_format_valid':sum(valid(r) for r in rows),'failed_started_episodes':sum(not valid(r) for r in rows),'not_started_episodes':192-len(rows),'full_repeat_blocks':full_blocks,'complete':complete,'original_order_prefix_verified':True,'all_started_initial_requests_and_contexts_verified':len(first),'training_cases':24,'confirmation_excluded':96,'frozen_selection_uses_question_only':True,'same_completed_case_results':observed,'planned_denominator_pairs':pairs,'missing_as_wrong_accounting_is_not_an_observed_scientific_comparison':True,'gate_passed':decision['pass'],'controller_started_events':len(starts),'remote_controller_finished_events':len(ends),'local_subscription_execute_calls':read('private/call_count.json'),'local_completed_response_files':len(responses),'responses_status_counts':dict(Counter(r['status'] for r in responses)),'remote_controller_usage':dict(usage),'request_seconds_sum_completed_local':sum(r.get('seconds',0) for r in responses),'responses_with_visible_429':sum('429 Too Many Requests' in r.get('stderr','') for r in responses),'visible_retry_events_lower_bound':len(waits),'visible_backoff_seconds_lower_bound':sum(waits),'internal_attempts_not_fully_counted_by_logical_calls':True,'tool_counts':dict(Counter(e['name'] for e in tools)),'tool_status_counts':dict(Counter(e['status'] for e in tools)),'historical_image_boundary_rejections':sum('Image outside current case' in str(e.get('result','')) for e in tools),'initial_final_consistency':flips,'original_wall_seconds':finish['seconds_since_original_start'],'original_deadline_epoch':read('budget.json')['hard_deadline_epoch'],'terminal_record_epoch':finish['time'],'final_guard':read('RUN_GUARD.json'),'medical_validity_established':False,'sequential_learning_established':False}
    (root/'AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
