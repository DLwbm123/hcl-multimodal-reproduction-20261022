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

ARMS=['Advisory','Contract']
ROOT, PLAN, BUDGET=base.ROOT,base.PLAN,base.BUDGET
base.policy_context=lambda *args:(FIXED,[])


from admission import envelope
CACHED=json.loads((ROOT/'private/cached_reports.json').read_text())


def probe(item, arm, tools, state):
    cached=CACHED.get(item['id'],[])
    if [x['image_path'] for x in cached] != item['tool_eligible_images']:
        raise RuntimeError('Cached observation qualification mismatch')
    if not cached:return ''
    admitted=[]
    for x in cached:
        assert x['status']=='completed'
        admitted.append({'image_path':x['image_path'],'evidence':envelope(x['observation'],arm)})
    base.record('cached_evidence_admitted',{'observations':admitted,'historical_report_requests':len(cached),
        'new_report_requests':0})
    return ('\nCurrent-image observations from previously recorded real tool calls follow. '
            'Both sections are fallible. Review flags are not clinical labels. '
            'Do not infer absence from withheld content or temporal change from a single-image call.\n'+json.dumps(admitted))


class AdmittedReport:
    def __init__(self, tool):
        self.tool=tool
        self.name,self.description,self.args_schema=tool.name,tool.description,tool.args_schema
    def invoke(self,args):
        result=self.tool.invoke(args)
        if not isinstance(result,tuple) or result[1].get('analysis_status')!='completed':return result
        base.record('fresh_report_before_admission',{'args':args,'original_result':str(result)})
        return json.dumps(envelope(str(result[0]),base.CONTEXT['arm'])),result[1]


original_load=base.load_four

def load_admitted():
    tools,state,Agent,prompt=original_load()
    for tool in tools:
        if tool.name=='chest_xray_report_generator':tool.tool=AdmittedReport(tool.tool)
    return tools,state,Agent,prompt

base.load_four=load_admitted


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
    pairs={'denominator':len(eligible),'observed_pairs':len(observed_pairs),'gain':sum(r['arms']['Advisory']['benchmark_agreement'] is False and r['arms']['Contract']['benchmark_agreement'] is True for r in observed_pairs),
           'loss':sum(r['arms']['Advisory']['benchmark_agreement'] is True and r['arms']['Contract']['benchmark_agreement'] is False for r in observed_pairs)}
    return rows,{'planned_episodes':len(expected),'recorded_episodes':len(seen),'coverage_complete':set(seen)==expected,'metrics':metrics,'paired':pairs,
        'unknown_reference_groups':len(inputs)-len(targets),'independent_evaluation':False,'learned_policy':False,'clinical_validation':False,'cached_report_acquisition_requests_per_arm':80}


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
        valid={a:sum(e['arm']==a and e.get('status')=='completed' and e.get('predicted') in list('ABCDEF') for e in base.EPISODES) for a in ARMS}
        report['valid_final']=valid
        report['source_diagnostic_continuation_signal']=(fatal is None and report['coverage_complete'] and
            report['paired']['gain']>=1 and report['paired']['loss']==0 and valid['Contract']>=valid['Advisory'])
        report['candidate_accepted']=False
        base.save('private/paired_feedback.json',rows);base.save('REPORT.json',report);base.save('FINISHED.json',report)
    return 0 if fatal is None else 2


if __name__=='__main__':raise SystemExit(main())
