"""Offline audit of every frozen v6 cell; no inference or resampling."""
import ast
import collections
import json
import os
from pathlib import Path
import re


def main():
    root=Path(os.environ['R981_ROOT']);previous=Path(os.environ['R998_PREVIOUS'])
    read=lambda name:json.loads((root/name).read_text())
    finish=read('FINISHED.json');episodes=read('private/episodes.json');inputs=read('private/inputs.json');source=read('private/source_state.json');exposures=read('private/exposures.json');report=read('private/report_results.json')
    assert finish['fatal'] is None and len(episodes)==120
    assert all(x['status']=='completed' and x['predicted'] in list('ABCDEF') for x in episodes)
    assert len({x['case_id'] for x in inputs})==len(inputs)==24
    assert source==json.loads((previous/'private/states.json').read_text())['H_A']
    train_ids={x['id'] for x in json.loads((previous/'private/inputs.json').read_text()) if x['role']=='train'}
    assert all(x['internal_source']['task_id'] in train_ids for x in source['raw'])
    assert not set(x['id'] for x in inputs)&train_ids
    assert all(set(x)=={'id','case_id','stage','role','question','images'} for x in inputs)
    assert len(exposures)==120
    arms=list(finish['metrics']);by_input={x['id']:x for x in inputs};by_episode={i:x for i,x in enumerate(episodes)}
    from hcl.memory import ExperienceMemory,MemoryConfig
    from interface import select,render_views
    memory=ExperienceMemory(MemoryConfig(record_dir=str(root/'private/audit_memory'),raw_capacity_per_task=10,abstract_capacity_per_task=2,skill_enabled=False,anchor_selection_mode='off'))
    memory.restore(source);views=(memory.raw.retrieval_candidates(),memory.abstract.retrieval_candidates())
    semantic_checks=0
    for ex in exposures:
        row=by_episode[ex['episode']];arm=row['state'];item=by_input[row['id']]
        chosen=select(*views,item['question'],arm)
        assert ex['arm']==arm and ex['ids']==row['retrieved_ids']==[v['memory_id'] for v in chosen]
        assert ex['kinds']==[v['memory_kind'] for v in chosen]
        if arm.startswith('balanced'):assert ex['kinds']==['raw','abstract']
        rendered=render_views(chosen,source['raw'],arm)
        context='' if not rendered else '\nHistorical training experience (different cases, not findings or answers for the current case). Use only applicable guidance; verify against current images and question:\n'+json.dumps(rendered,ensure_ascii=False)
        assert len(context)==row['context_chars']
        semantic_checks+=arm.endswith('_semantic')
    assert memory.snapshot()==source
    events=[json.loads(x) for x in (root/'private/events.jsonl').read_text().splitlines()]
    calls=[x for x in events if x['kind']=='controller_finished'];starts=[x for x in events if x['kind']=='controller_started'];tools=[x for x in events if x['kind']=='tool_finished'];usage=collections.Counter();flags=collections.Counter()
    assert len(calls)==len(starts)==361 and all(x['status']=='ok' for x in calls)
    assert all(x['images']>0 and x['phase']=='case' for x in starts)
    assert min(x['time'] for x in starts)>read('FROZEN.json')['time']
    for x in calls:
        for u in x.get('usage',[]):usage.update(u)
    for x in tools:
        value=ast.literal_eval(x['result'])[0];text=value.get('response','') if isinstance(value,dict) and 'response' in value else value if isinstance(value,str) else None
        if text is None:continue
        if not text.strip():flags['empty']+=1
        if text.strip().lower()=='not matched':flags['not_matched']+=1
        if text.strip().lower()=='no':flags['bare_no']+=1
        if 'bibibibibibi' in text:flags['repetitive_malformed']+=1
    per_arm={}
    for arm in arms:
        rows=[x for x in episodes if x['state']==arm];assert len(rows)==24 and len({x['id'] for x in rows})==24
        assert sum(x['correct'] for x in report[arm])==sum(finish['metrics'][arm][s]['correct'] for s in ['A','B'])
        bare=[x for x in rows if x['initial_final'].strip() in list('ABCDEF')]
        expanded=[(x,re.match(r'([A-F])(?:[).:\s]|$)',x['initial_final'].lstrip().lstrip('*'))) for x in rows]
        expanded=[(x,m) for x,m in expanded if m]
        per_arm[arm]={'initial_mean_chars':sum(len(x['initial_final']) for x in rows)/24,'bare_initial_letters_omitted_by_frozen_secondary_parser':len(bare),'supplemental_explicit_initial_parseable':len(expanded),'supplemental_initial_final_changes':sum(m.group(1)!=x['predicted'] for x,m in expanded),'tools':dict(collections.Counter(x['name'] for x in tools if x['state']==arm)),'context_min_chars':min(x['context_chars'] for x in rows),'context_max_chars':max(x['context_chars'] for x in rows)}
    value={'complete_audit_passed':True,'physical_episodes':120,'unique_report_case_groups':24,'all_inputs_answer_free':True,'source_exactly_prior_frozen_training_state':True,'actual_retrieval_and_context_reconstructed':120,'semantic_rendering_reconstructed':semantic_checks,'source_unchanged':True,'source_frozen_before_inference':True,'controller_calls':len(calls),'controller_calls_with_images':len(starts),'controller_usage':dict(usage),'controller_request_seconds_sum':sum(x.get('seconds',0) for x in calls),'tool_counts':dict(collections.Counter(x['name'] for x in tools)),'tool_statuses':dict(collections.Counter(x['status'] for x in tools)),'tool_output_surface_flags':dict(flags),'surface_flags_are_not_medical_correctness_labels':True,'per_arm':per_arm,'final_guard':read('RUN_GUARD.json'),'seconds_since_original_start':finish['seconds_since_original_start'],'combined_development_signal':False,'limitations':['Single fixed memory snapshot and 24 development groups; no stochastic replication, new acquisition or retention test','Frozen secondary regex excludes a bare single letter at end of text; retain original metrics and label expanded parser supplemental','Abstract guidance includes output-form instructions; exposure changes observed response format, not a pure retrieval-quality contrast','Contexts are slot-matched but not token-matched; all factor comparisons retained','No arm improves pooled accuracy over H0; this fixed interface family is closed without winner adoption']}
    (root/'AUDIT.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))

if __name__=='__main__':main()
