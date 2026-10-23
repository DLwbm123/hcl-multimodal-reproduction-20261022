"""No-model, aggregate-only audit of a failed capability gate and names control."""
import argparse,json,sys,collections
from pathlib import Path

def main(root,repo,deps):
    root=Path(root);sys.dont_write_bytecode=True
    sys.path[:0]=[str(root/'code'),repo,deps]
    from strict_scoring import score_prediction
    from clinical_catalog import identity
    from medharness.scorers.calc import answer_episode_ids
    visible=json.loads((root/'data/visible_split.json').read_text())
    gold=json.loads((root/'data/evaluator_gold.json').read_text())
    learning={c['id'] for c in visible if c['role'].startswith('train_')}
    assert len(learning)==8
    diagnostic=json.loads((root/'DIAGNOSTIC.json').read_text())
    assert not (root/'FORMAL_STARTED.json').exists()
    assert not json.loads((root/'ACCEPTANCE.json').read_text())['passed']
    report={'formal_started':False,'acceptance_passed':False,'phases':{}}
    total=0
    for phase,key in [('acceptance_unguided','unguided'),('diagnostic_oracle_names','oracle_names')]:
        records=[json.loads(p.read_text()) for p in (root/phase).glob('*/RESULT.json')]
        assert len(records)==8 and {r['case_id'] for r in records}==learning
        errors=collections.Counter();end=collections.Counter();tools=collections.Counter();executed=cached_execution=dups=0
        unmatched=collections.Counter();calc_calls=0
        for r in records:
            expected=score_prediction(r['answer'] or {'calculators':[]},gold[r['case_id']])
            assert expected==r['score']
            end[r['termination']]+=1;dups+=r['duplicates_cached'];calc_calls+=r['model_calls']
            predictions=(r['answer'] or {}).get('calculators',[])
            for a,ep,row in zip(gold[r['case_id']]['calculator_answers'],answer_episode_ids(gold[r['case_id']]),r['score']['rows']):
                if row['selected']:continue
                names=[p for p in predictions if identity(p.get('name',''))==identity(a['name'])]
                exact=[p for p in names if str(p.get('episode_id',''))==str(ep)]
                unmatched['no_name_in_submission' if not names else ('wrong_episode' if not exact else 'duplicate_identity_episode')]+=1
            for t in r['trace']:
                if t['role']!='tool':continue
                tools[t['name']]+=1
                try:content=json.loads(t['content'])
                except ValueError:content={}
                if t.get('status')=='error':
                    detail=content.get('detail',t['content'])
                    cat=('malformed_json' if 'malformed tool call' in detail else
                         'unknown_calculator_id' if 'unknown calculator_id' in detail else
                         'disabled_formula' if 'not enabled' in detail else
                         'missing_parameter' if 'required property' in detail else
                         'wrong_parameter_type' if 'not of type' in detail else 'other')
                    errors[cat]+=1
                elif t['name']=='execute_calculator':
                    if content.get('cached'):cached_execution+=1
                    else:executed+=1
        s=diagnostic[key]
        assert sum(r['score']['n_correct'] for r in records)==s['correct']
        assert sum(len(r['score']['rows']) for r in records)==s['outputs']==67
        assert sum(r['submitted'] for r in records)==s['submitted']
        assert sum(errors.values())==s['tool_errors']
        assert sum(tools.values())==s['tool_calls']
        assert executed+cached_execution==s['execution_successes']
        report['phases'][key]={'model_generations':calc_calls,'termination_counts':dict(end),'tool_call_counts':dict(tools),'error_categories':dict(errors),'unmatched_output_categories':dict(unmatched),'duplicates_cached':dups,'formula_executions_uncached':executed,'formula_success_cached':cached_execution}
        total+=calc_calls
    calls=[json.loads(x) for x in (root/'model_calls.jsonl').read_text().splitlines()]
    attempts=json.loads((root/'attempts.json').read_text())['calls']
    assert total==len(calls)==attempts==283
    assert sorted(c['call'] for c in calls)==list(range(1,284))
    pc=[json.loads(x) for x in (root/'protocol_calls.jsonl').read_text().splitlines()]
    invalid=[p['call'] for p in pc if p['invalid']]
    assert invalid==[216,218,221]
    report['cost']={'attempts':attempts,'real_generations':len(calls),'generation_seconds':sum(c['seconds'] for c in calls),'input_tokens':sum(c['input_tokens'] for c in calls),'output_tokens':sum(c['output_tokens'] for c in calls),'invalid_protocol_calls':len(invalid),'invalid_output_tokens':[c['output_tokens'] for c in calls if c['call'] in invalid]}
    report['exit_codes']={n:json.loads((root/(n+'.json')).read_text())['code'] for n in ['EXIT','SUPERVISOR_EXIT']}
    assert set(report['exit_codes'].values())=={2}
    report['checks']=['same eight learning cases only','stored scoring reproduced without model calls','67 outputs per condition','tool/error accounting','cached execution separated','all 283 actual generations accounted','formal gate not bypassed','expected exit 2']
    (root/'COMPLETION_AUDIT.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':main(*sys.argv[1:])
