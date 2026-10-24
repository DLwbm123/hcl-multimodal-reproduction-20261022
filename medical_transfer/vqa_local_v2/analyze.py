"""Completed-run audit and aggregate-only paired analysis; never selects a model."""
import collections,json,os
from pathlib import Path

ORDER=['A','B','C']
CHECKPOINTS=['empty']+ORDER
ARMS=['direct','static','fixed_memory','plasticity','stability','bulk']

def paired(before,after):
    assert set(before)==set(after) and before
    n=len(before);lost=sum(before[k] and not after[k] for k in before)
    gained=sum(not before[k] and after[k] for k in before);correct=sum(before.values())
    return {'n':n,'before_correct':correct,'after_correct':sum(after.values()),'lost':lost,'gained':gained,
            'delta_pp':100*(gained-lost)/n,'loss_per_all_percent':100*lost/n,
            'loss_per_reference_correct_percent':100*lost/correct if correct else None}

def interval(strata,seed=2026102402):
    import numpy as np
    rng=np.random.default_rng(seed);boot=np.zeros(20000)
    for values in strata:
        v=np.asarray(values,dtype=float);assert len(v)>0
        boot+=v[rng.integers(0,len(v),size=(20000,len(v)))].mean(axis=1)/len(strata)
    return [float(x) for x in np.quantile(boot*100,[.025,.975])]

def compare(left,right):
    assert len(left)==len(right)
    ds=[];parts=[]
    for a,b in zip(left,right):
        assert set(a)==set(b)
        ids=sorted(a);ds.append([int(a[k])-int(b[k]) for k in ids]);parts.append(paired(b,a))
    return {'difference_pp':sum(p['delta_pp'] for p in parts)/len(parts),'paired95_pp':interval(ds),'strata':parts}

def load_rows(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]

def analyze(root):
    root=Path(root);plan=json.loads((root/'protocol.json').read_text());budget=json.loads((root/'budget.json').read_text())
    supervisor=json.loads((root/'SUPERVISOR_EXIT.json').read_text())
    assert supervisor['returncodes']=={'stability':0,'plasticity':0,'bulk':0,'controls':0} and not supervisor.get('failed')
    assert not (root/'HARD_STOP.json').exists()
    data=json.loads((root/'private_data.json').read_text());index={};cost={};exits={};audit={}
    all_examples=[e for stage in data.values() for xs in stage.values() for e in xs]
    assert len(all_examples)==504 and len({e['metadata']['group_id'] for e in all_examples})==504
    assert len({e['metadata']['pixel_id'] for e in all_examples})==504
    assert all(e['task_name']=='visual_qa' and e['metadata']['role']!='test' for e in all_examples)
    for worker in ['stability','plasticity','bulk','controls']:
        out=root/worker;exit_info=json.loads((out/'EXIT.json').read_text());exits[worker]=exit_info
        assert exit_info['exit_code']==0 and not (out/'FAILURE.txt').exists()
        assert exit_info['ended']<=budget['hard_deadline_epoch']
        calls=load_rows(out/'call_costs.jsonl');attempts=json.loads((out/'model_attempts.json').read_text())['attempts']
        assert len(calls)==attempts==exit_info['calls'] and [c['attempt'] for c in calls]==list(range(1,attempts+1))
        assert attempts<=plan['budget']['max_calls_controls' if worker=='controls' else 'max_calls_per_hcl']
        assert all(c['success'] for c in calls)
        # Final image payload is an execution requirement, separate from answer correctness.
        finals=[c for c in calls if c['phase']=='final_answer'];assert finals and all(c['image_supplied'] for c in finals)
        scopes=collections.defaultdict(lambda:{'calls':0,'seconds':0.0,'peak_allocated_bytes':0})
        for c in calls:
            s=c['scope'].split(':')[0];category='report_evaluation' if s=='report' else 'anchor_audit' if s=='anchor' else 'learning_and_selection'
            bucket=scopes[category];bucket['calls']+=1;bucket['seconds']+=c['seconds'];bucket['peak_allocated_bytes']=max(bucket['peak_allocated_bytes'],c['peak_allocated_bytes'])
        cost[worker]=dict(scopes)
        blocks=load_rows(out/'private_scores.jsonl');readonly=load_rows(out/'read_only_checks.jsonl')
        assert len(readonly)==len(blocks)
        checks={(x['arm'],x['checkpoint'],x['stage'],x['kind']):x for x in readonly}
        predictions=collections.Counter((x['arm'],x['checkpoint'],x['stage'],x['kind']) for x in load_rows(out/'private_predictions.jsonl'))
        for x in blocks:
            key=(x['arm'],x['checkpoint'],x['stage'],x['kind']);assert key not in index
            expected={e['task_id'] for e in data[x['stage']][x['kind']]}
            assert len(x['rows'])==len(expected) and {y['task_id'] for y in x['rows']}==expected
            assert predictions[key]==len(expected) and x['metrics']['total']==len(expected)
            assert sum(y['success'] for y in x['rows'])==x['metrics']['correct']
            assert checks[key]['unchanged'] and checks[key]['weights_frozen'] and checks[key]['count']==len(expected)
            index[key]=x
        audit[worker]={'scored_blocks':len(blocks),'readonly_blocks':len(readonly),'actual_answers':sum(predictions.values()),'final_calls_with_image':len(finals),'model_calls':attempts}
    for arm in ['direct','static']:
        for checkpoint in ORDER:
            for stage in ORDER:index[(arm,checkpoint,stage,'report')]=index[(arm,'empty',stage,'report')]
    def correctness(arm,checkpoint,stage,kind='report'):
        return {x['task_id']:bool(x['success']) for x in index[(arm,checkpoint,stage,kind)]['rows']}
    result={'protocol':plan['name'],'complete':True,'audit':audit,'cost':cost,'total_model_calls':sum(e['calls'] for e in exits.values()),
            'started_epoch':budget['started_epoch'],'ended_epoch':supervisor['ended'],
            'elapsed_seconds':supervisor['ended']-budget['started_epoch'],'final_new_bytes_with_reserve':supervisor['bytes'],
            'matrix':{},'learning':{},'retention':{},'anchors':{},'updates':{},'comparisons':{}}
    assert result['final_new_bytes_with_reserve']<plan['budget']['storage_cap_bytes']
    for arm in ARMS:
        result['matrix'][arm]={c:{s:{'correct':index[(arm,c,s,'report')]['metrics']['correct'],
                                  'valid':index[(arm,c,s,'report')]['metrics']['format_compliant_count'],'n':72} for s in ORDER} for c in CHECKPOINTS}
        gains=[]
        for j,s in enumerate(ORDER):
            before=correctness(arm,CHECKPOINTS[j],s);after=correctness(arm,s,s)
            gain=paired(before,after);gain['paired95_pp']=compare([after],[before])['paired95_pp'];gains.append(gain)
        result['learning'][arm]={'by_stage':dict(zip(ORDER,gains)),'mean_immediate_gain_pp':sum(x['delta_pp'] for x in gains)/3}
        retention={}
        for checkpoint in ['B','C']:
            previous=CHECKPOINTS[CHECKPOINTS.index(checkpoint)-1]
            for s in ORDER[:ORDER.index(checkpoint)]:
                now=correctness(arm,checkpoint,s)
                retention[f'{checkpoint}_on_{s}']={'since_previous':paired(correctness(arm,previous,s),now),'since_acquisition':paired(correctness(arm,s,s),now)}
        result['retention'][arm]=retention
        result['retention'][arm]['final_bwt_pp']=sum(retention[f'C_on_{s}']['since_acquisition']['delta_pp'] for s in ['A','B'])/2
        if arm in ['direct','static']:continue
        result['anchors'][arm]={}
        for checkpoint in ORDER:
            for s in ORDER[:ORDER.index(checkpoint)+1]:
                current=correctness(arm,checkpoint,s,'anchor');ref=correctness(arm,s,s,'anchor')
                by={x['task_id']:x for x in data[s]['anchor']};coverage=collections.defaultdict(lambda:{'total':0,'acquired_correct':0,'current_correct':0,'lost':0,'gained':0})
                for k in current:
                    row=coverage[by[k]['metadata']['question_type']];row['total']+=1;row['acquired_correct']+=int(ref[k]);row['current_correct']+=int(current[k]);row['lost']+=int(ref[k] and not current[k]);row['gained']+=int(not ref[k] and current[k])
                result['anchors'][arm][f'{checkpoint}_on_{s}']={**paired(ref,current),'type_coverage':dict(coverage)}
    for arm in ['stability','plasticity','bulk']:
        rows=load_rows(root/arm/'update_decisions.jsonl');assert len(rows)==6
        for i,row in enumerate(rows):
            assert row['stage']==ORDER[i//2] and row['batch']==i%2+1 and row['arm']==arm
            assert len(row['candidates'])==3
            chosen=row['chosen_index'];assert row['accepted']==(chosen is not None)
            for candidate in row['candidates']:
                assert candidate['raw_count']<=10 and candidate['abstract_count']<=2
                if candidate['eligible']:
                    cur=candidate['metrics']['current_task_metrics'];old=candidate['metrics']['historical_anchor_metrics']
                    assert cur['correct']>row['reference']['current_task_metrics']['correct'] and cur['format_compliance_rate']==1
                    if arm!='plasticity':assert old.get('forget_count',0)==0
            if chosen is not None:assert row['candidates'][chosen]['eligible']
        result['updates'][arm]={'memory':rows,'selector':[]}
    for left,right in [('stability','plasticity'),('stability','static'),('stability','fixed_memory'),('plasticity','static'),('plasticity','fixed_memory'),('stability','bulk')]:
        key=left+'_minus_'+right
        result['comparisons'][key]={c:{'macro':compare([correctness(left,c,s) for s in ORDER],[correctness(right,c,s) for s in ORDER]),
                                           'by_stage':{s:compare([correctness(left,c,s)],[correctness(right,c,s)]) for s in ORDER}} for c in ORDER}
    # Per-case forgetting uses each arm's acquired-correct reference. Also disclose
    # the common-acquired-correct subset so an unlearned stage cannot look protected.
    loss_diff=[];common={}
    for s in ['A','B']:
        acq={a:correctness(a,s,s) for a in ['stability','plasticity']};final={a:correctness(a,'C',s) for a in acq}
        ids=sorted(acq['stability']);loss_diff.append([int(acq['stability'][k] and not final['stability'][k])-int(acq['plasticity'][k] and not final['plasticity'][k]) for k in ids])
        both=[k for k in ids if acq['stability'][k] and acq['plasticity'][k]]
        common[s]={'n':len(both),'stability_lost':sum(not final['stability'][k] for k in both),'plasticity_lost':sum(not final['plasticity'][k] for k in both)}
    loss={'difference_pp':100*sum(sum(v)/len(v) for v in loss_diff)/2,'paired95_pp':interval(loss_diff),'common_acquired_correct':common}
    result['historical_loss_stability_minus_plasticity']=loss
    mean_gain=result['learning']['stability']['mean_immediate_gain_pp'];p_gain=result['learning']['plasticity']['mean_immediate_gain_pp']
    static=result['comparisons']['stability_minus_static']['C']['macro'];fixed=result['comparisons']['stability_minus_fixed_memory']['C']['macro']
    later_update=all(any(x['accepted'] and x['stage']==stage for x in result['updates']['stability']['memory']) and result['learning']['stability']['by_stage'][stage]['delta_pp']>0 for stage in ['B','C'])
    retention_ok=all(result['retention']['stability'][f'C_on_{s}']['since_acquisition']['lost']<=3 and result['retention']['stability'][f'C_on_{s}']['since_acquisition']['before_correct']>0 for s in ['A','B'])
    p_lost=sum(result['retention']['plasticity'][f'C_on_{s}']['since_acquisition']['lost'] for s in ['A','B'])
    result['frozen_gates']={'learning_and_retention':static['difference_pp']>=5 and static['paired95_pp'][0]>0 and mean_gain>=5 and later_update and retention_ok,
                          'gate_attribution':p_lost>0 and loss['difference_pp']<0 and loss['paired95_pp'][1]<0 and mean_gain>=p_gain-5,
                          'adaptive_advantage_over_fixed_memory':fixed['paired95_pp'][0]>0,
                          'components':{'final_vs_static_pp':static['difference_pp'],'mean_immediate_gain_pp':mean_gain,'accepted_later_update':later_update,'old_stage_loss_limits_passed':retention_ok,'mean_learning_gap_vs_plasticity_pp':mean_gain-p_gain}}
    bulk=result['comparisons']['stability_minus_bulk']['C']['macro']
    losses=lambda arm:sum(result['retention'][arm][f'C_on_{s}']['since_acquisition']['lost'] for s in ['A','B'])
    locality=bulk['paired95_pp'][0]>0 and losses('stability')<=losses('bulk') and mean_gain>=result['learning']['bulk']['mean_immediate_gain_pp']
    result['frozen_gates']['locality']=locality
    result['frozen_gates']['combined']=all(result['frozen_gates'][k] for k in ['learning_and_retention','adaptive_advantage_over_fixed_memory','locality'])
    result['data_scope']={'reused_development_groups':288,'new_report_groups':216,'original_report_role':'train_replay','report_role':'reserved_report','original_test_used':False,'patient_independent':False,'paired_with_v1':False}
    return result

if __name__=='__main__':
    root=Path(os.environ['RUN_ROOT']);result=analyze(root)
    (root/'ANALYSIS.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({'complete':result['complete'],'gates':result['frozen_gates'],'calls':result['total_model_calls']}))
