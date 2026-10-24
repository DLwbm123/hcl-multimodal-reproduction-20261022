"""Offline aggregate audit; no model calls or private text in public output."""
import ast
import collections
import json
import math
import os
from pathlib import Path


def aggregate(root):
    finish=json.loads((root/'FINISHED.json').read_text())
    rows=json.loads((root/'private/results.json').read_text())
    events=[json.loads(x) for x in (root/'private/events.jsonl').read_text().splitlines()]
    prior=[json.loads(x) for x in (root/'attempts/path_namespace_01/private/events.jsonl').read_text().splitlines()]
    assert len(rows)==12 and len({r['id'] for r in rows})==12
    assert finish['completed']==sum(r['status']=='completed' for r in rows)
    starts=[e for e in events if e['kind']=='controller_started' and e['phase']=='native_panel']
    calls=[e for e in events if e['kind']=='controller_finished' and e['phase']=='native_panel']
    tool=[e for e in events if e['kind']=='tool_finished' and e['phase']=='native_panel']
    assert len(calls)==finish['controller_calls']==36 and len(tool)==12
    flags=collections.Counter();usage=collections.Counter();rewrites=0
    for e in tool:
        result=ast.literal_eval(e['result'])[0]
        text=result.get('response','') if isinstance(result,dict) else result
        if not text.strip():flags['empty']+=1
        if text.strip().lower()=='not matched':flags['not_matched']+=1
        if text.strip().lower()=='no':flags['bare_no']+=1
        if 'bibibibibibi' in text:flags['repetitive_malformed']+=1
    phases={}
    for label,es in [('initial_interrupted',prior),('repaired_complete',events)]:
        cs=[e for e in es if e['kind']=='controller_finished'];u=collections.Counter()
        for e in cs:
            for item in e.get('usage',[]):u.update(item)
        phases[label]={'controller_started':sum(e['kind']=='controller_started' for e in es),'controller_finished':len(cs),'usage':dict(u),'controller_seconds_sum':sum(e.get('seconds',0) for e in cs),'tool_statuses':dict(collections.Counter(e['status'] for e in es if e['kind']=='tool_finished' and e['phase']=='native_panel'))}
    responses=list((root/'private/queue').glob('*.response.json'))
    for f in responses:rewrites+=len(json.loads(f.read_text()).get('transport_path_rewrites',[]))
    n=len(rows);correct=sum(r['correct'] for r in rows);p=correct/n;z=1.959963984540054;den=1+z*z/n
    center=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return {'denominator':n,'completed':finish['completed'],'valid_final':finish['valid_final'],'correct':correct,'accuracy':p,'wilson_95_descriptive':[center-half,center+half],'all_six_preflight_passed':all(r['success'] for r in finish['preflight']),'autonomous_tool_calls':len(tool),'autonomous_tool_counts':dict(collections.Counter(e['name'] for e in tool)),'successful_tool_episodes':finish['episodes_with_successful_tool'],'tool_output_surface_flags':dict(flags),'surface_flags_are_not_medical_validity_judgments':True,'all_controller_requests_have_images':all(e['images']>0 for e in starts),'image_attachment_total':sum(e['images'] for e in starts),'completed_requests_with_tool_results':sum(e['tool_results']>0 for e in starts),'exact_attachment_alias_rewrites':rewrites,'hcl_updates':0,'operational_qualified':finish['operational_qualified'],'medical_or_hcl_efficacy_established':False,'attempt_costs':phases,'seconds_since_original_start_to_finish':finish['time']-json.loads((root/'budget.json').read_text())['started_epoch'],'final_guard_bytes':json.loads((root/'RUN_GUARD.json').read_text())['bytes'],'interpretation':'Development qualification only. No paired no-tool control, no independent confirmation, no causal tool-benefit or HCL claim. All failures and retry costs retained.'}

if __name__=='__main__':
    root=Path(os.environ['R981_ROOT']);value=aggregate(root)
    (root/'AGGREGATE_REPORT.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value))
