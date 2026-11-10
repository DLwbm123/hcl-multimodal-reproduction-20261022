"""Source-only paired action feedback; no candidate is deployed by this run."""
import json
import re
import signal
import subprocess
import time
import traceback
from pathlib import Path
import base_worker as base
from qualified_tool import install_shared_guard
from policy import FIXED

ARMS=['Fixed','Probe']
ROOT, PLAN, BUDGET=base.ROOT,base.PLAN,base.BUDGET
base.policy_context=lambda *args:(FIXED,[])


def probe(item, arm, tools, state):
    if arm!='Probe':return ''
    expert=next(t for t in tools if t.name=='chest_xray_expert')
    stem=re.split(r'\n\s*A[).]\s',item['question'],maxsplit=1)[0]
    query=('Describe only visible radiographic findings relevant to the following question. '
           'Do not choose an answer option or infer disease cause from absence of a finding. '
           'If the requested feature cannot be established, say uncertain. Question: '+stem)
    observations=[]
    for path in item['tool_eligible_images']:
        base.guard()
        before=len(base.EVENTS)
        result=expert.invoke({'image_paths':[path],'prompt':query,'max_new_tokens':256})
        status=result[1].get('analysis_status') if isinstance(result,tuple) else 'unknown'
        if 'Visual encoding contract failed' in str(result):
            raise RuntimeError('Visual-contract invariant breached')
        observations.append({'image_path':path,'status':status,'observation':str(result[0]),
                             'event_offset':before,'decision_source':'frozen_exploration_action'})
    base.record('paired_probe',{'image_count':len(item['images']),'observations':observations})
    return ('\nThe following are actual fallible observations from existing tools on the current images. '
            'Each observation is bound to its image; a failed request is unknown, never absence. '
            'These do not establish etiology, chronology, severity, or the final choice. '
            'Use only relevant supported observations and retain current-image reasoning.\n'+json.dumps(observations))


base.episode_probe=probe


def score(inputs, episodes, targets):
    expected={(x['id'],a) for x in inputs for a in ARMS};seen=[(e['id'],e['arm']) for e in episodes]
    assert set(targets)=={x['id'] for x in inputs if x['semantic_feedback_allowed']}
    assert len(seen)==len(set(seen)) and set(seen)<=expected
    by={(e['id'],e['arm']):e for e in episodes};rows=[]
    for item in inputs:
        pair={a:by.get((item['id'],a),{}) for a in ARMS}
        feedback={a:({'benchmark_agreement':pair[a].get('status')=='completed' and pair[a].get('predicted')==targets[item['id']]} if item['id'] in targets else {'benchmark_agreement':None}) for a in ARMS}
        rows.append({'id':item['id'],'question':item['question'],'image_count':len(item['images']),
            'composite_image_count':len(item['composite_image_files']),'semantic_feedback_allowed':item['semantic_feedback_allowed'],
            'reference_tier':item['reference_tier'],'arms':{a:{'status':pair[a].get('status','missing'),'predicted':pair[a].get('predicted'),
                'tool_calls':pair[a].get('tool_calls',0),'successful_tools':pair[a].get('successful_tools',0),
                'error':pair[a].get('error'),**feedback[a]} for a in ARMS}})
    eligible=[r for r in rows if r['semantic_feedback_allowed']]
    metrics={a:{'planned':len(inputs),'recorded':sum(e['arm']==a for e in episodes),'completed':sum(e['arm']==a and e['status']=='completed' for e in episodes),
        'semantic_denominator':len(targets),'benchmark_correct':sum(r['arms'][a]['benchmark_agreement'] is True for r in eligible),
        'tool_calls':sum(e['tool_calls'] for e in episodes if e['arm']==a),'seconds':sum(e['seconds'] for e in episodes if e['arm']==a)} for a in ARMS}
    observed_pairs=[r for r in eligible if all(r['arms'][a]['status']!='missing' for a in ARMS)]
    pairs={'denominator':len(eligible),'observed_pairs':len(observed_pairs),'gain':sum(r['arms']['Fixed']['benchmark_agreement'] is False and r['arms']['Probe']['benchmark_agreement'] is True for r in observed_pairs),
           'loss':sum(r['arms']['Fixed']['benchmark_agreement'] is True and r['arms']['Probe']['benchmark_agreement'] is False for r in observed_pairs)}
    return rows,{'planned_episodes':len(expected),'recorded_episodes':len(seen),'coverage_complete':set(seen)==expected,'metrics':metrics,'paired':pairs,
        'unknown_reference_groups':len(inputs)-len(targets),'independent_evaluation':False,'learned_policy':False,'clinical_validation':False}


def main():
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('Independent supervisor stop')))
    inputs=json.loads((ROOT/'private/inputs.json').read_text())
    assert len(inputs)==46 and len({x['source_key'] for x in inputs})==46
    assert sum(x['semantic_feedback_allowed'] for x in inputs)==13
    assert all(x['privacy_approved'] and all(Path(p).is_file() for p in x['images']) for x in inputs)
    base.save('STARTED.json',{'time':time.time(),'original_start':BUDGET['started_epoch'],'planned_episodes':92})
    import os
    free={int(x.split(',')[0]):int(x.split(',')[1]) for x in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()}
    assert free[int(os.environ['CUDA_VISIBLE_DEVICES'])]>=45000
    fatal=None
    try:
        base.CONTEXT['phase']='loading_native_tools'
        tools,state,Agent,prompt=base.load_four();install_shared_guard(tools,state,base.record)
        base.save('TOOLS_LOADED.json',{'time':time.time(),'names':[t.name for t in tools]})
        base.rt.SubscriptionModel().load()
        for i,item in enumerate(inputs):
            for arm in (ARMS if i%2==0 else list(reversed(ARMS))):
                base.episode(item,arm,tools,state,Agent,prompt,None,None)
    except BaseException as exc:
        fatal=repr(exc);base.save('FAILED.json',{'error':fatal,'traceback':traceback.format_exc(),'context':base.CONTEXT})
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        targets=json.loads((ROOT/'private/qualified_targets.json').read_text())
        assert set(targets)=={x['id'] for x in inputs if x['semantic_feedback_allowed']}
        rows,report=score(inputs,base.EPISODES,targets)
        report.update(fatal=fatal,elapsed_since_original_start=time.time()-BUDGET['started_epoch'])
        base.save('private/paired_feedback.json',rows);base.save('REPORT.json',report);base.save('FINISHED.json',report)
    return 0 if fatal is None else 2


if __name__=='__main__':raise SystemExit(main())
