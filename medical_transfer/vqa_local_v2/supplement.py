"""Post-hoc sensitivity/descriptive analyses; never changes frozen gates or runtime."""
import collections,json,math,os
from pathlib import Path
from analyze import ARMS,ORDER,CHECKPOINTS,load_rows


def four_cells(left,right):
    assert left and set(left)==set(right)
    both=sum(left[k] and right[k] for k in left)
    lonly=sum(left[k] and not right[k] for k in left)
    ronly=sum(right[k] and not left[k] for k in left)
    n=lonly+ronly
    p=min(1.,2*sum(math.comb(n,k) for k in range(min(lonly,ronly)+1))/2**n) if n else 1.
    return {'n':len(left),'both_correct':both,'both_wrong':len(left)-both-lonly-ronly,'left_only_correct':lonly,'right_only_correct':ronly,'discordant':n,'exact_two_sided_mcnemar_p':p,'informative_discordant_pairs':bool(n)}


def retained(group,current):
    n=len(group);correct=sum(current[k] for k in group)
    return {'reference_count':n,'still_correct':correct,'lost':n-correct,'retained_fraction':correct/n if n else None,'applicable':bool(n)}


def analyze_supplement(root):
    root=Path(root);primary=json.loads((root/'ANALYSIS.json').read_text())
    assert primary['complete'] and primary['protocol']=='vqa_local_v2'
    done=json.loads((root/'SUPERVISOR_EXIT.json').read_text())
    assert done['returncodes']=={'stability':0,'plasticity':0,'bulk':0,'controls':0} and not done.get('failed')
    data=json.loads((root/'private_data.json').read_text())
    meta={e['task_id']:e['metadata'] for s in data.values() for xs in s.values() for e in xs}
    index={};predictions=[]
    for worker in ['stability','plasticity','bulk','controls']:
        for x in load_rows(root/worker/'private_scores.jsonl'):
            key=x['arm'],x['checkpoint'],x['stage'],x['kind'];assert key not in index;index[key]=x
        predictions+=load_rows(root/worker/'private_predictions.jsonl')
    for arm in ['direct','static']:
        for c in ORDER:
            for s in ORDER:index[arm,c,s,'report']=index[arm,'empty',s,'report']
    def correct(arm,c,s):return {x['task_id']:bool(x['success']) for x in index[arm,c,s,'report']['rows']}
    result={'status':'complete_supplementary_post_hoc','changes_frozen_gates':False,'multiple_testing_adjustment':False,
            'interpretation':'Exact tests are supplementary sensitivity analyses, not replacement success criteria. All comparisons disclosed; content-group independence is not patient independence. F1 does not establish visual correctness.',
            'type_metrics':{},'new_knowledge_retention':{},'paired_tables':{},'candidate_reasons':{},'memory_occupancy':{},'report_retrieval':{}}
    for arm in ARMS:
        result['type_metrics'][arm]={};result['new_knowledge_retention'][arm]={}
        for c in CHECKPOINTS:
            bytype={}
            for s in ORDER:
                for qtype in sorted({meta[x['task_id']]['question_type'] for x in index[arm,c,s,'report']['rows']}):
                    xs=[x for x in index[arm,c,s,'report']['rows'] if meta[x['task_id']]['question_type']==qtype]
                    bytype[qtype]={'n':len(xs),'correct':sum(x['success'] for x in xs),'valid':sum(x['valid'] for x in xs),'answer_f1_diagnostic':sum(x['auxiliary'] for x in xs)/len(xs)}
            result['type_metrics'][arm][c]=bytype
        for i,s in enumerate(ORDER):
            before=correct(arm,CHECKPOINTS[i],s);after=correct(arm,s,s);base=correct(arm,'empty',s)
            groups={'newly_correct_this_stage':{k for k in before if not before[k] and after[k]},'correct_before_this_stage':{k for k in before if before[k]},'correct_in_initial_empty_state':{k for k in base if base[k]},'all_correct_at_stage_end':{k for k in after if after[k]}}
            result['new_knowledge_retention'][arm][s]={c:{name:retained(ids,correct(arm,c,s)) for name,ids in groups.items()} for c in ORDER[i:]}
            result['paired_tables'][f'{arm}:{s}:immediate']=four_cells(after,before)
        if arm in ['direct','static']:continue
        result['memory_occupancy'][arm]={}
        worker='controls' if arm=='fixed_memory' else arm
        for c in ORDER:
            state=json.loads((root/worker/f'private_committed_{arm}_{c}.json').read_text())['memory']
            raw_by_type=collections.Counter(meta[x['internal_source']['task_id']]['question_type'] for x in state['raw'])
            result['memory_occupancy'][arm][c]={'raw':len(state['raw']),'abstract':len(state['abstract']),'raw_question_types':dict(raw_by_type)}
    for a,b in [('stability','plasticity'),('stability','bulk'),('stability','fixed_memory'),('stability','static'),('plasticity','fixed_memory')]:
        for c in ORDER:
            left={};right={}
            for s in ORDER:
                l=correct(a,c,s);rr=correct(b,c,s);assert not set(left)&set(l);left.update(l);right.update(rr)
            result['paired_tables'][f'{a}_minus_{b}:{c}']={'pooled':four_cells(left,right),'by_stage':{s:four_cells(correct(a,c,s),correct(b,c,s)) for s in ORDER}}
    for arm in ['stability','plasticity','bulk']:
        decisions=load_rows(root/arm/'update_decisions.jsonl');assert len(decisions)==6
        result['candidate_reasons'][arm]={}
        for s in ORDER:
            ds=[x for x in decisions if x['stage']==s];candidates=[x for d in ds for x in d['candidates']]
            reasons=collections.Counter(reason for x in candidates for reason in x['reasons'])
            combinations=collections.Counter('|'.join(x['reasons']) or 'eligible' for x in candidates)
            result['candidate_reasons'][arm][s]={'batches':len(ds),'accepted_updates':sum(x['accepted'] for x in ds),'candidates':len(candidates),'reason_counts_overlapping':dict(reasons),'exact_reason_combinations':dict(combinations)}
    buckets=collections.defaultdict(lambda:{'answers':0,'selected_total':0,'empty_selection':0,'selected_kinds':collections.Counter(),'selected_ids':set()})
    for x in predictions:
        if x['kind']!='report':continue
        trace=x['prediction'].get('trace',{});items=trace.get('selected_memory',[])
        key=':'.join([x['arm'],x['checkpoint'],x['stage']]);b=buckets[key]
        b['answers']+=1;b['selected_total']+=len(items);b['empty_selection']+=int(not items)
        for item in items:
            b['selected_kinds'][item.get('memory_kind','unknown')]+=1
            if item.get('memory_id'):b['selected_ids'].add(item['memory_id'])
    for key,b in buckets.items():
        result['report_retrieval'][key]={k:v for k,v in b.items() if k not in ['selected_ids','selected_kinds']}
        result['report_retrieval'][key].update(selected_kinds=dict(b['selected_kinds']),unique_selected_records=len(b['selected_ids']))
    result['memory_flow_evidence_limits']={'available':'committed stage-end occupancy/type composition, candidate bank sizes/decisions, report-time retrieval counts/kinds; no new model calls',
        'missing':'Per-candidate proposal snapshots and all 24 intermediate raw states were not explicitly persisted by the v2 worker. Do not claim exact reconstruction of the full train-to-proposal-to-two-record chain from these aggregate tables.',
        'development_selection_trace_followup':'Inspect existing private development traces for P1 separately; do not substitute report errors for a new independent development selection set.'}
    return result


def self_check():
    cases=[(7,0,.015625),(9,2,.0654296875),(4,0,.125),(0,0,1.)]
    for gained,lost,expected in cases:
        left={str(i):i<gained for i in range(gained+lost+2)}
        right={str(i):gained<=i<gained+lost for i in range(gained+lost+2)}
        x=four_cells(left,right);assert x['left_only_correct']==gained and x['right_only_correct']==lost
        assert abs(x['exact_two_sided_mcnemar_p']-expected)<1e-12
        assert sum(x[k] for k in ['both_correct','both_wrong','left_only_correct','right_only_correct'])==x['n']
    assert retained(set(),{})['retained_fraction'] is None
    assert retained({'a','b'},{'a':True,'b':False})['retained_fraction']==.5
    try:four_cells({'a':True},{'b':True})
    except AssertionError:pass
    else:raise AssertionError('identity mismatch accepted')
    return {'passed':True,'checks':['exact_discrete_pairs_and_zero_discordance','four_cell_denominators','empty_new_learning_is_not_applicable','partial_retention','mismatched_identities_rejected'],'model_calls':0}

if __name__=='__main__':
    root=Path(os.environ['RUN_ROOT']);self_check();result=analyze_supplement(root)
    (root/'SUPPLEMENT.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({'complete':True,'supplementary':True,'frozen_gates_changed':False,'model_calls':0}))
