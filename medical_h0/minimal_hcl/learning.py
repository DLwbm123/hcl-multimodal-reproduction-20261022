"""Restricted HCL: official optimizer gate, automatic workflow proposals, atomic commits."""
import json,re
from pathlib import Path
from hcl_core.optimizer import Optimizer
from tools import schema

CANDIDATE_SCHEMA=schema('propose_workflow','Propose one reusable workflow instruction artifact, not an executable tool or an answer.',{
 'candidate_name':{'type':'string','minLength':1,'maxLength':80},
 'rationale':{'type':'string','maxLength':1000},
 'sections':{'type':'array','minItems':1,'maxItems':5,'items':{'type':'string','minLength':1,'maxLength':400}},
 'artifact_diff_summary':{'type':'string','maxLength':500}},['candidate_name','rationale','sections','artifact_diff_summary'])

def atomic(path,value):
    path=Path(path);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)

def validate(sections):
    if not isinstance(sections,list) or not 1<=len(sections)<=5 or any(not isinstance(s,str) or not 1<=len(s)<=400 for s in sections):return ['invalid_sections']
    text='\n'.join(sections)
    # This intentionally narrow v1 permits procedural prose only. It rejects
    # numeric examples, placeholders and executable/code-bearing instructions.
    if re.search(r'[0-9{}]|https?://|/data/|__|```|\b(exec|eval|import|open|shell|subprocess|os\.)\b',text,re.I):return ['nonprocedural_or_executable_content']
    if re.search(r'ignore (the |all |previous )?(system|rules|instructions)|disable|bypass|hidden answers|change.*(schema|scorer|budget)',text,re.I):return ['boundary_override']
    return []

def optimizer(root):
    return Optimizer(record_dir=Path(root)/'optimizer',candidate_count=2,components=['medical_workflow'],
      historical_loss_budget=0,min_correct_gain=1,min_primary_score_delta=0.,min_format_compliance_rate=1.,
      min_historical_anchor_count=4,min_historical_anchors_per_task=4,require_all_historical_tasks=True,
      min_anchor_evaluation_coverage=1.,screening_ratio=1.,screening_min_current_count=4,
      screening_min_anchors_per_task=4,selection_mode='argmax')

def metrics(results):
    return {'correct':sum(r['success'] for r in results),'total':len(results),
      'primary_score':sum(r['utility'] for r in results)/len(results),
      'format_compliance_rate':sum(r['submitted'] for r in results)/len(results)}

def paired(before,after):
    assert [r['case_id'] for r in before]==[r['case_id'] for r in after]
    return {'total':len(before),'forget_count':sum(a['success'] and not b['success'] for a,b in zip(before,after)),
      'recovered_count':sum(not a['success'] and b['success'] for a,b in zip(before,after)),
      'baseline_success_count':sum(r['success'] for r in before)}

def choose(opt,candidates,evaluations,current,anchors):
    results={}
    for c in candidates:
        cid=c['candidate_id'];ev=evaluations.get(cid)
        if ev is None:
            results[cid]={'metrics':{'invalid_candidate':True}}
            continue
        assert len(ev['current'])==4 and len(ev['anchors'])==len(anchors)
        results[cid]={'current_task_metrics':metrics(ev['current']),
            'historical_anchor_metrics':paired(anchors,ev['anchors'])}
    return opt.select_best_candidate(candidates,results,{'current_task_metrics':metrics(current)})

def commit(path,current,candidate,stage):
    value=dict(current)
    if candidate:
        value={'version':current['version']+1,'sections':candidate['artifact_content']['sections'],
               'candidate_id':candidate['candidate_id'],'stage':stage}
    atomic(path,value)
    assert json.loads(Path(path).read_text())==value
    return value
