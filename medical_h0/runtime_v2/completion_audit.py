"""CPU-only post-run audit; no generation and no changes to learned state."""
import json,sys
from pathlib import Path

def audit(root):
    root=Path(root);result=json.loads((root/'RESULT.json').read_text())
    records=[json.loads(x) for x in (root/'evaluations.jsonl').read_text().splitlines()]
    visible=json.loads((root/'data/visible_split.json').read_text())
    roles={c['id']:c['role'] for c in visible}
    assert len(roles)==32
    assert result['complete']
    static=[r for r in records if r['phase']=='final_static'];hcl=[r for r in records if r['phase']=='final_hcl']
    assert len(static)==len(hcl)==8 and [r['case_id'] for r in static]==[r['case_id'] for r in hcl]
    assert all(roles[r['case_id']].startswith('confirm_') for r in static)
    assert all(not roles[r['case_id']].startswith('confirm_') for r in records if not r['phase'].startswith('final_'))
    calls=[json.loads(x) for x in (root/'model_calls.jsonl').read_text().splitlines()]
    attempts=json.loads((root/'attempts.json').read_text())['calls']
    assert attempts==result['model_calls']
    missing=sorted(set(range(1,attempts+1))-{c['call'] for c in calls})
    assert missing==[135],missing
    failures=[json.loads(line) for path in (root/'episodes').glob('*/*/events.jsonl') for line in path.read_text().splitlines() if json.loads(line).get('event')=='model_call_failed']
    assert len(failures)==1 and 'context length budget' in failures[0]['error']
    assert len(calls)+len(missing)==attempts
    private_pairs={}
    for stage,task in [(1,'dependency'),(2,'submission')]:
        saved=json.loads((root/f'stage{stage}.json').read_text())
        generation=json.loads((root/f'generation_input_s{stage}.json').read_text())
        allowed={c['prompt'] for c in visible if c['role']=='train_'+task}
        assert len(generation['training'])==4 and {t['prompt'] for t in generation['training']}==allowed
        before=[r for r in records if r['phase']==f's{stage}_anchors']
        assert len(before)==(4 if stage==1 else 8)
        for label in sorted({r['phase'] for r in records if r['phase'].startswith(f's{stage}_candidate') and r['phase'].endswith('_anchors')}):
            after=[r for r in records if r['phase']==label]
            assert [r['case_id'] for r in before]==[r['case_id'] for r in after]
            private_pairs[label]=[{'case_id':a['case_id'],'before_success':a['success'],'after_success':b['success'],'lost':a['success'] and not b['success'],'outputs_before':a['output_ok'],'outputs_after':b['output_ok']} for a,b in zip(before,after)]
        if saved['decision']['accepted']:
            accepted=saved['decision'];assert accepted['anchor_forget_count']==0 and accepted['correct_gain']>=1
            assert saved['after']['version']==saved['before']['version']+1
        else:assert saved['after']==saved['before']
    (root/'anchor_transitions.json').write_text(json.dumps(private_pairs,indent=2))
    protocol=[json.loads(x) for x in (root/'protocol_calls.jsonl').read_text().splitlines()]
    report={'complete':True,'evaluated_episodes':len(records),'model_calls':len(calls),'model_attempts':attempts,'pre_generation_refusals':len(missing),'acceptance_generations':20,'formal_generations':len(calls)-20,'generation_seconds':sum(c['seconds'] for c in calls),'input_tokens':sum(c['input_tokens'] for c in calls),'output_tokens':sum(c['output_tokens'] for c in calls),'invalid_tool_calls':sum(c['invalid'] for c in protocol),'confirmation_cases_per_arm':8,'checks':['role isolation','fixed paired confirmation','candidate generation train-only prompts','full anchor pairs','commit/rollback state','call accounting']}
    (root/'COMPLETION_AUDIT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':audit(sys.argv[1])
