"""Read-only endpoint audit. No inference, source rewriting or score changes."""
import ast
from collections import Counter
import json
import math
import os
from pathlib import Path
import re
from qualification import first_errors, source_summary, valid


def paired(a, b, ids, gold):
    counts = Counter((a[k]['predicted'] == gold[k], b[k]['predicted'] == gold[k]) for k in ids)
    gain, loss = counts[False, True], counts[True, False]
    n = gain + loss
    return dict(both_correct=counts[True, True], gain=gain, loss=loss,
                both_wrong=counts[False, False],
                mcnemar_exact_two_sided=min(1., 2 * sum(math.comb(n, i) for i in range(min(gain, loss)+1)) / 2**n), denominator=len(ids))


def main():
    root = Path(os.environ['R981_ROOT'])
    read = lambda name: json.loads((root/name).read_text())
    report = read('FINISHED.json'); rows = read('private/episodes.json')
    inputs = read('private/inputs.json'); gold = read('private/supervision.json')
    members = read('private/members.json'); selected = read('private/error_recheck_members.json')
    assert report['complete'] and report['fatal'] is None
    assert len(rows) == 104 and all(valid(r) for r in rows)
    assert len(inputs) == len({r['case_id'] for r in inputs}) == len(gold) == 72
    assert {r['id'] for r in inputs} == set(gold)
    assert all(set(r) == {'id','case_id','stage','role','question','images'} for r in inputs)
    reserve = [r for r in members if r['role'] == 'confirmation_reserved']
    assert len(reserve) == 96 and not {r['full_question_id'] for r in reserve} & set(gold)
    assert not {str(r['case_id']) for r in reserve} & {str(r['case_id']) for r in inputs}
    assert all(not r['retrieved_ids'] and r['context_chars'] == 0 for r in rows)
    by = {label: {r['id']:r for r in rows if r['state']==label} for label in report['initial_final_consistency']}
    repeats = [r for r in inputs if r['role']=='repeat_development']
    train = [r for r in inputs if r['role']=='train']
    expected = []
    for stage in ['A','B']:
        ids = [r['id'] for r in repeats if r['stage']==stage]
        assert paired(by['H0_repeat_1'],by['H0_repeat_2'],ids,gold) == report['repeatability'][stage]['correctness_paired_full_denominator']
        items = [r for r in train if r['stage']==stage]
        chosen = first_errors(items,list(by['H0_train'].values()),gold,4); expected.extend(chosen)
        assert source_summary(items,list(by['H0_train'].values()),list(by['H0_error_recheck'].values()),chosen,gold) == report['sources'][stage]
    assert [r['id'] for r in expected] == [r['id'] for r in selected]
    planned = [('H0_repeat_1',r['id']) for r in repeats] + [('H0_repeat_2',r['id']) for r in reversed(repeats)] + [('H0_train',r['id']) for r in train] + [('H0_error_recheck',r['id']) for r in expected]
    assert [(r['state'],r['id']) for r in rows] == planned
    for label, summary in report['initial_final_consistency'].items():
        entries=list(by[label].values()); parsed=[re.match(r'([A-F])(?:[).:\s]|$)',r['initial_final'].lstrip().lstrip('*')) for r in entries]
        assert all(parsed) and summary['initial_parseable_valid']==len(entries)
        assert sum(m.group(1)!=r['predicted'] for m,r in zip(parsed,entries))==summary['initial_final_changed']
    events=[json.loads(s) for s in (root/'private/events.jsonl').read_text().splitlines()]
    starts=[r for r in events if r['kind']=='controller_started']; calls=[r for r in events if r['kind']=='controller_finished']; tools=[r for r in events if r['kind']=='tool_finished']
    assert len(starts)==len(calls)==327 and {r['key'] for r in starts}=={r['key'] for r in calls}
    assert all(r['status']=='ok' for r in calls) and all(r['images']>0 for r in starts)
    assert min(r['time'] for r in starts)>read('FROZEN.json')['time']
    assert max(r['time'] for r in calls if r['state']=='H0_train') < read('RECHECKS_FROZEN.json')['time'] < min(r['time'] for r in starts if r['state']=='H0_error_recheck')
    usage=Counter(); flags=Counter(); trajectories={i:[] for i in range(len(rows))}
    for r in calls:
        for u in r.get('usage',[]):usage.update(u)
    for r in tools:
        trajectories[r['episode']].append(r['name'])
        value=ast.literal_eval(r['result'])[0]
        text=value.get('response','') if isinstance(value,dict) and 'response' in value else value if isinstance(value,str) else None
        if text is not None:
            if not text.strip():flags['empty']+=1
            if text.strip().lower()=='not matched':flags['not_matched']+=1
            if text.strip().lower()=='no':flags['bare_no']+=1
            if 'bibibibibibi' in text:flags['repetitive_malformed']+=1
    indexes={(r['state'],r['id']):i for i,r in enumerate(rows)}
    repeat_tools={s:sum(trajectories[indexes['H0_repeat_1',r['id']]]!=trajectories[indexes['H0_repeat_2',r['id']]] for r in repeats if r['stage']==s) for s in ['A','B']}
    result={'complete_audit_passed':True,'physical_episodes':len(rows),'unique_case_inputs':len(inputs),'empty_contexts_checked':len(rows),'deterministic_first_error_selection_checked':True,'all_paired_and_source_tables_recomputed':True,'execution_order_checked':True,'reserved_confirmation_ids_excluded_from_inputs_and_supervision':96,'controller_calls':len(calls),'controller_calls_with_images':len(starts),'controller_usage':dict(usage),'reasoning_tokens_are_subset_not_additional':True,'controller_request_seconds_sum':sum(r['seconds'] for r in calls),'tool_counts':dict(Counter(r['name'] for r in tools)),'tool_statuses':dict(Counter(r['status'] for r in tools)),'tool_output_surface_flags':dict(flags),'surface_flags_not_medical_labels':True,'repeat_pairs_with_different_ordered_tool_names':repeat_tools,'episode_seconds_by_state':{label:sum(r['seconds'] for r in entries.values()) for label,entries in by.items()},'configured_model':read('budget.json')['model'],'configured_reasoning':read('budget.json')['reasoning'],'backend_snapshot_identity_available':False,'original_wall_seconds':report['seconds_since_original_start'],'final_guard':read('RUN_GUARD.json'),'medical_source_validity_established':False,'continual_learning_established':False}
    (root/'AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
