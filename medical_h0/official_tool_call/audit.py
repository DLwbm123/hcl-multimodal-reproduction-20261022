"""Aggregate completed synthetic receipts without any model calls."""
import json
from pathlib import Path


def audit(root):
    root=Path(root)
    assert "Ran 13 tests" in (root/"CPU_TEST_NOUS.log").read_text() and "\nOK" in (root/"CPU_TEST_NOUS.log").read_text()
    scripted=json.loads((root/"cpu/RESULT.json").read_text())
    assert scripted["passed"] and scripted["cases"]==6 and scripted["model_calls"]==0
    arms={}
    for folder,label in [('real','legacy_qwen_compatibility'),('real_nous','qwen25vl_nous')]:
        p=root/folder
        receipt=json.loads((p/'RESULT.json').read_text())
        calls=[json.loads(line) for line in (p/'model_calls.jsonl').read_text().splitlines()]
        protocol=[json.loads(line) for line in (p/'protocol_calls.jsonl').read_text().splitlines()]
        assert len(calls)==receipt['model_calls']==len(protocol)
        rows=[]
        for result in receipt['results']:
            rows.append({'case':result['case_id'],'passed':result['passed'],
                'termination':result['termination'],'requests':result['requests'],
                'utility':result['score']['utility'],'correct_outputs':result['score']['n_correct'],
                'scored_outputs':result['score']['n_scored'],
                'injected_error':result['injected_error'],'tool_errors':result['tool_errors'],
                'submitted':result['answer'] is not None})
        arms[label]={'passed':receipt['passed'],'episodes':rows,
            'actual_generations':len(calls),'parsed_valid_calls':sum(x['valid'] for x in protocol),
            'parsed_invalid_calls':sum(x['invalid'] for x in protocol),
            'generation_seconds_including_load':sum(x['seconds'] for x in calls),
            'input_tokens':sum(x['input_tokens'] for x in calls),
            'output_tokens':sum(x['output_tokens'] for x in calls),
            'exit_code':json.loads((p/'EXIT.json').read_text())['code'],
            'supervisor_exit_code':json.loads((p/'SUPERVISOR_EXIT.json').read_text())['code']}
    return {'protocol':'official_fncall_v1','gate_passed':arms['qwen25vl_nous']['passed'],
        'cpu_tests':{'passed':13,'total':13},'scripted_graph_cases':6,
        'actual_generations_total':sum(a['actual_generations'] for a in arms.values()),
        'new_hcl_runs':0,'os_sandbox':False,'benchmark_cases_read':0,
        'scope':'Synthetic engineering acceptance only; no clinical benchmark or CL result.',
        'censoring':'Final Nous recovery episode stopped at shared 144-call cap; provider_failure is budget exhaustion, not a malformed response. It is not a completed negative accuracy observation.',
        'arms':arms}

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);a=parser.parse_args()
    result=audit(a.root)
    (a.root/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
