"""Offline completed-run audit. Outputs aggregates only; no additional inference."""
import ast
import collections
import json
import os
from pathlib import Path
import re


def main():
    root=Path(os.environ['R981_ROOT']);finish=json.loads((root/'FINISHED.json').read_text())
    assert finish['fatal'] is None
    episodes=json.loads((root/'private/episodes.json').read_text());inputs=json.loads((root/'private/inputs.json').read_text());supervision=json.loads((root/'private/supervision.json').read_text())
    assert len(episodes)==160 and all(x['status']=='completed' and x['predicted'] in list('ABCDEF') for x in episodes)
    assert len({x['case_id'] for x in inputs})==88
    train_ids={x['id'] for x in inputs if x['role']=='train'};report_ids={x['id'] for x in inputs if x['role']=='report'}
    assert not report_ids&supervision.keys()
    states=json.loads((root/'private/states.json').read_text());assert states['H_AB_local_zero']==states['H_AB_local_infinite']==states['H_A']
    assert all(r['internal_source']['task_id'] in train_ids for s in states.values() for r in s['raw'])
    events=[json.loads(x) for x in (root/'private/events.jsonl').read_text().splitlines()]
    calls=[x for x in events if x['kind']=='controller_finished'];starts=[x for x in events if x['kind']=='controller_started'];usage=collections.Counter()
    for x in calls:
        for u in x.get('usage',[]):usage.update(u)
    assert len(calls)==len(starts)==488 and all(x['status']=='ok' for x in calls)
    tools=[x for x in events if x['kind']=='tool_finished'];flags=collections.Counter()
    for x in tools:
        value=ast.literal_eval(x['result'])[0];text=value.get('response','') if isinstance(value,dict) and 'response' in value else value if isinstance(value,str) else None
        if text is None:continue
        if not text.strip():flags['empty']+=1
        if text.strip().lower()=='not matched':flags['not_matched']+=1
        if text.strip().lower()=='no':flags['bare_no']+=1
        if 'bibibibibibi' in text:flags['repetitive_malformed']+=1
    flips=[];parseable=0
    for x in episodes:
        m=re.match(r'([A-F])[).:\s]',x.get('initial_final','').lstrip().lstrip('*'))
        if m:
            parseable+=1
            if m.group(1)!=x['predicted']:flips.append(x)
    retrieval={}
    for name in sorted({x['state'] for x in episodes}):
        rows=[x for x in episodes if x['state']==name];ids=[i for r in rows for i in r['retrieved_ids']]
        retrieval[name]={'episodes':len(rows),'raw_retrievals':sum(i.startswith('raw_') for i in ids),'abstract_retrievals':sum(i.startswith('abstract_') for i in ids),'distinct_retrieved_memories':len(set(ids)),'context_chars_mean':sum(r['context_chars'] for r in rows)/len(rows)}
    train={s:{'correct':sum(x['predicted']==supervision[x['id']] for x in episodes if x['state']==s+'_train'),'denominator':8} for s in ['A','B']}
    candidate_stats={}
    for stage in ['A','B']:
        c=json.loads((root/('private/candidate_'+stage+'.json')).read_text());p=json.loads((root/('private/proposal_'+stage+'.json')).read_text())
        candidate_stats[stage]={'raw_count':len(c['raw']),'abstract_count':len(c['abstract']),'retained_raw_count':sum(r in c['raw'] for r in states['H_A']['raw']) if stage=='B' else 0,'all_raw_sources_train':all(r['internal_source']['task_id'] in train_ids for r in p['raw']),'training_snapshots':len(list((root/'private').glob('training_memory_'+stage+'_*.json')))}
    report_events=[x for x in events if x.get('state') in {'H0_static','H_A'}]
    freeze=json.loads((root/'STATES_FROZEN.json').read_text());assert min(x['time'] for x in report_events)>freeze['time']
    value={'complete_audit_passed':True,'physical_episodes':160,'unique_case_groups':88,'unique_report_groups':48,'report_matrix_named_cells':192,'physical_report_episodes':96,'controller_calls':488,'controller_usage':dict(usage),'controller_request_seconds_sum':sum(x.get('seconds',0) for x in calls),'abstract_calls':sum(x['phase']=='abstract_proposal' for x in starts),'case_calls_with_images':sum(x['phase']=='case' and x['images']>0 for x in starts),'case_calls_total':sum(x['phase']=='case' for x in starts),'tool_counts':dict(collections.Counter(x['name'] for x in tools)),'tool_statuses':dict(collections.Counter(x['status'] for x in tools)),'tool_output_surface_flags':dict(flags),'surface_flags_are_not_medical_correctness_labels':True,'training_accuracy':train,'candidate_write_audit':candidate_stats,'retrieval':retrieval,'initial_explicit_letter_parseable':parseable,'second_turn_letter_changes':len(flips),'letter_changes_by_state':dict(collections.Counter(x['state'] for x in flips)),'report_states_frozen_before_predictions':True,'report_labels_absent_from_supervision':True,'all_raw_sources_from_training':True,'final_guard':json.loads((root/'RUN_GUARD.json').read_text()),'seconds_since_original_start':finish['seconds_since_original_start'],'joint_development_signal':finish['joint_development_signal'],'gate_contribution_pattern':finish['gate_contribution_pattern'],'limitations':['A acquired-correct report set empty; retention of newly learned A not established','B selection starts at 8/8, so required +1 gain is unattainable in this frozen panel; candidate also loses one','Abstract guidance generated but never retrieved; observed behavior is effectively two raw examples','One second-turn letter change causes the sole A report decline; keep original second-turn scoring','No stochastic repeat and no causal attribution of letter copying from this single observed change']}
    (root/'AUDIT.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))

if __name__=='__main__':main()
