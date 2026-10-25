"""Offline full-run audit; never performs additional inference."""
import ast
import collections
import json
import os
from pathlib import Path
import re
from banks import ARMS,EMPTY,matched_banks


def main():
    root=Path(os.environ['R981_ROOT']);read=lambda n:json.loads((root/n).read_text())
    finish=read('FINISHED.json');episodes=read('private/episodes.json');inputs=read('private/inputs.json');supervision=read('private/supervision.json');states=read('private/states.json');pool=read('private/training_pool.json');reports=read('private/report_results.json');gold=read('private/sealed_report_gold.json')
    assert finish['fatal'] is None and len(episodes)==176 and all(x['status']=='completed' and x['predicted'] in list('ABCDEF') for x in episodes)
    assert len(inputs)==len({x['case_id'] for x in inputs})==80
    train=[x for x in episodes if x['role']=='train'];report=[x for x in episodes if x['role']=='report'];assert len(train)==32 and len(report)==144
    train_ids={x['id'] for x in inputs if x['role']=='train'};report_ids={x['id'] for x in inputs if x['role']=='report'}
    assert set(supervision)==train_ids and set(gold)==report_ids and not train_ids&report_ids
    assert all(set(x)=={'id','case_id','stage','role','question','images'} for x in inputs)
    assert len(pool['raw'])==32 and {x['internal_source']['task_id'] for x in pool['raw']}==train_ids
    assert len(list((root/'private').glob('training_memory_*.json')))==32
    rebuilt,audit=matched_banks(pool['raw'],inputs,train,supervision);assert rebuilt==states and audit==finish['training_bank_audit']
    assert all(not s['abstract'] and not s['anchor'] and not s['skills'] for s in states.values())
    assert len(states['corrective_memory']['raw'])==len(states['success_memory']['raw'])==2
    from hcl.memory import ExperienceMemory,MemoryConfig
    from worker import context
    memory=ExperienceMemory(MemoryConfig(record_dir=str(root/'private/audit_memory'),raw_capacity_per_task=32,abstract_capacity_per_task=2,skill_enabled=False,anchor_selection_mode='off'))
    items={x['id']:x for x in inputs};per_arm={}
    for row in episodes:
        snapshot=EMPTY if row['role']=='train' else states[row['state']];memory.restore(snapshot)
        text,ids=context(memory,items[row['id']]['question'],train_ids)
        assert ids==row['retrieved_ids'] and len(text)==row['context_chars'] and memory.snapshot()==snapshot
    for name in ARMS:
        rows=reports[name];assert len(rows)==48 and {x['id'] for x in rows}==report_ids
        assert all(x['correct']==(x['predicted']==gold[x['id']]) for x in rows)
        for stage in ['A','B']:assert sum(x['correct'] for x in rows if x['stage']==stage)==finish['metrics'][name][stage]['correct']
        raw=states[name]['raw'];per_arm[name]={'bank_raw_records':len(raw),'retrieved_raw_reads':sum(len(x['retrieved_ids']) for x in rows),'distinct_retrieved_records':len({i for x in rows for i in x['retrieved_ids']}),'context_chars_mean':sum(x['context_chars'] for x in rows)/48,'all_bank_sources_B_training':all(items[x['internal_source']['task_id']]['stage']=='B' for x in raw) if raw else None}
    events=[json.loads(x) for x in (root/'private/events.jsonl').read_text().splitlines()];calls=[x for x in events if x['kind']=='controller_finished'];starts=[x for x in events if x['kind']=='controller_started'];tools=[x for x in events if x['kind']=='tool_finished'];usage=collections.Counter();flags=collections.Counter()
    assert len(starts)==len(calls)==536 and all(x['status']=='ok' for x in calls) and all(x['images']>0 for x in starts)
    assert min(x['time'] for x in starts)>read('FROZEN.json')['time']
    freeze=read('STATES_FROZEN.json')['time'];assert max(x['time'] for x in calls if x['state']=='training_H0')<freeze<min(x['time'] for x in starts if x['state'] in ARMS)
    for x in calls:
        for u in x.get('usage',[]):usage.update(u)
    for x in tools:
        value=ast.literal_eval(x['result'])[0];text=value.get('response','') if isinstance(value,dict) and 'response' in value else value if isinstance(value,str) else None
        if text is None:continue
        if not text.strip():flags['empty']+=1
        if text.strip().lower()=='not matched':flags['not_matched']+=1
        if text.strip().lower()=='no':flags['bare_no']+=1
        if 'bibibibibibi' in text:flags['repetitive_malformed']+=1
    result={'complete_audit_passed':True,'physical_episodes':176,'unique_cases':80,'unique_report_cases':48,'all_training_and_report_contexts_reconstructed':176,'banks_exactly_reconstructed_from_training_only':True,'training_snapshots':32,'training_labels_only_before_report':True,'all_reports_after_states_frozen':True,'all_states_immutable':True,'training_bank_audit':audit,'per_arm':per_arm,'controller_calls':536,'controller_calls_with_images':536,'controller_usage':dict(usage),'controller_request_seconds_sum':sum(x.get('seconds',0) for x in calls),'tool_counts':dict(collections.Counter(x['name'] for x in tools)),'tool_statuses':dict(collections.Counter(x['status'] for x in tools)),'tool_output_surface_flags':dict(flags),'surface_flags_are_not_medical_correctness_labels':True,'seconds_since_original_start':finish['seconds_since_original_start'],'final_guard':read('RUN_GUARD.json'),'source_error_coverage':False,'acquisition_development_signal':False,'limitations':['No A training errors, so no A corrective record; cannot test A correction learning or infer A acquisition','Only two B raw records in each bank; every report receives both, with no effective relevance filtering','Corrective versus H0 gain1 loss0 falls below the frozen at-least2 threshold; vs success gain2 includes reversal of one success-bank regression','No stochastic repeat or confirmatory significance; all frozen endpoints and denominators retained','Question/image/answer source content differs between error and success banks; equal record count is not equal context length','Acquisition signal not established; sequential learning and retention untested']}
    (root/'AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
