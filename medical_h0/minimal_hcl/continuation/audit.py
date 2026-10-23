"""CPU-only post-run audit; no generation and no changes to learned state."""
import json,sys
from pathlib import Path

def audit(root):
    root=Path(root).resolve();result=json.loads((root/'RESULT.json').read_text())
    records=[json.loads(x) for x in (root/'evaluations.jsonl').read_text().splitlines()]
    old_records=[json.loads(x) for x in (root.parent/'evaluations.jsonl').read_text().splitlines()]
    reused=sum(r.get('reused_from_initial_run',False) for r in records)
    assert reused==15
    records=[r for r in old_records if r['phase'].startswith('s1_')]+records
    visible=json.loads((root/'data/visible_split.json').read_text())
    roles={c['id']:c['role'] for c in visible}
    assert len(roles)==36
    assert result['complete']
    static=[r for r in records if r['phase']=='final_static'];hcl=[r for r in records if r['phase']=='final_hcl']
    assert len(static)==len(hcl)==12 and [r['case_id'] for r in static]==[r['case_id'] for r in hcl]
    assert all(roles[r['case_id']].startswith('confirm_') for r in static)
    assert all(not roles[r['case_id']].startswith('confirm_') for r in records if not r['phase'].startswith('final_'))
    calls=[json.loads(x) for x in (root/'model_calls.jsonl').read_text().splitlines()]
    attempts=json.loads((root/'attempts.json').read_text())['calls']
    assert len(calls)==result['model_calls'] and attempts==283+len(calls)
    old_calls=[json.loads(x) for x in (root.parent/'model_calls.jsonl').read_text().splitlines()]
    assert len(old_calls)==282
    calls=old_calls+calls
    private_pairs={}
    for stage,task in [(1,'dependency'),(2,'submission')]:
        saved=json.loads((root/f'stage{stage}.json').read_text())
        generation=json.loads(((root.parent if stage==1 else root)/f'generation_input_s{stage}.json').read_text())
        allowed={c['prompt'] for c in visible if c['role']=='train_'+task}
        assert len(generation['training'])==4 and {t['prompt'] for t in generation['training']}==allowed
        before=[r for r in records if r['phase']==f's{stage}_anchors']
        assert len(before)==(4 if stage==1 else 8)
        for label in sorted({r['phase'] for r in records if r['phase'].startswith(f's{stage}_candidate') and r['phase'].endswith('_anchors')}):
            after=[r for r in records if r['phase']==label]
            assert [r['case_id'] for r in before]==[r['case_id'] for r in after]
            private_pairs[label]=[{'case_id':a['case_id'],'before_success':a['success'],'after_success':b['success'],'lost':a['success'] and not b['success']} for a,b in zip(before,after)]
        if saved['decision']['accepted']:
            accepted=saved['decision'];assert accepted['anchor_forget_count']==0 and accepted['correct_gain']>=1
            assert saved['after']['version']==saved['before']['version']+1
        else:assert saved['after']==saved['before']
    (root/'anchor_transitions.json').write_text(json.dumps(private_pairs,indent=2))
    protocol=[json.loads(x) for base in [root.parent,root] for x in (base/'protocol_calls.jsonl').read_text().splitlines()]
    report={'complete':True,'evaluated_episodes':len(records),'model_calls':len(calls),'attempted_calls':attempts,'original_interrupted_calls':1,'reused_stage2_episodes':reused,'new_completed_calls':result['model_calls'],'generation_seconds':sum(c['seconds'] for c in calls),'input_tokens':sum(c['input_tokens'] for c in calls),'output_tokens':sum(c['output_tokens'] for c in calls),'invalid_tool_calls':sum(c['invalid'] for c in protocol),'confirmation_cases_per_arm':12,'checks':['role isolation','fixed paired confirmation','candidate generation train-only prompts','full anchor pairs','commit/rollback state','call accounting']}
    (root/'AUDIT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':audit(sys.argv[1])
