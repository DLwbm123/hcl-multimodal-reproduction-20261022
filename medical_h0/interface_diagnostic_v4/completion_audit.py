import json,sys,time
from pathlib import Path

def audit(root):
 r=Path(root);result=json.loads((r/'RESULT.json').read_text());attempts=json.loads((r/'attempts.json').read_text())['calls'];calls=[json.loads(x) for x in (r/'model_calls.jsonl').read_text().splitlines()];protocol=[json.loads(x) for x in (r/'protocol_calls.jsonl').read_text().splitlines()];budget=json.loads((r/'budget.json').read_text());previous=budget.get('previous_model_calls',0)
 recs=[json.loads(p.read_text()) for p in sorted((r/'episodes').glob('*/*/RESULT.json'))]
 assert attempts-previous==len(calls)==len(protocol)==sum(x['model_calls'] for x in recs)
 assert all(json.loads((r/n).read_text())['code'] in (0,2) for n in ['EXIT.json','SUPERVISOR_EXIT.json'])
 assert not any((r/n).exists() for n in ['FAILURE.json','HARD_STOP.json'])
 selected=json.loads((r/'D0_SELECTION.private.json').read_text())['items'];assert len(selected)==6
 invalid=[call for x in recs for t in x['trace'] if t['role']=='assistant' for call in t.get('invalid_tool_calls',[])]
 out={'complete':True,'attempts_this_run':attempts-previous,'actual_generations':len(calls),'previous_attempts':previous,'total_attempts':attempts,'input_tokens':sum(x['input_tokens'] for x in calls),'output_tokens':sum(x['output_tokens'] for x in calls),'generation_seconds':sum(x['seconds'] for x in calls),'invalid_model_calls':len(invalid),'length_capped_calls':sum(x['output_tokens']>=4096 for x in calls),'episodes':len(recs),'hcl_started':False,'data_scope':'six preselected targets from original learning cases, assisted diagnostics only','layers':result['layers']}
 (r/'COMPLETION_AUDIT.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
if __name__=='__main__':audit(sys.argv[1])
