"""Persist fit-only follow-up credits separately from the frozen agent's memory."""
import json
import sys
from pathlib import Path
from hcl.memory.raw import RawMemory


def collect(policy_root, output, shards):
    training = json.loads((policy_root/'private_fit.json').read_text())
    expected = {r['id']:r for r in training if not r['quote']}
    rows = {}
    for shard in shards:
        assert json.loads((shard/'SUPERVISOR_EXIT.json').read_text())['success']
        part = json.loads((shard/'private_filtered.json').read_text())
        assert rows.keys().isdisjoint(part)
        rows.update(part)
    assert rows.keys()==expected.keys()
    output.mkdir(exist_ok=True)
    examples=[]
    supported=0
    for k,r in rows.items():
        item=dict(cue=expected[k]['cue'], before=r['pre'], after=r['post'], tested_sentences=r['phrases'])
        positive = any(p['only']==1 and p['without']==0 for p in r['phrases'])
        supported+=positive
        examples.append(dict(task_id=k,question=json.dumps(item),response='feature-supported',
                             answer='feature-supported' if positive else 'unconfirmed',split='fit',task_type='feature_check'))
    raw=RawMemory(output/'raw.jsonl',capacity_per_task=len(rows))
    raw.add('filtered_fit_followup',examples,[dict(task_id=e['task_id'],answer=e['response']) for e in examples])
    assert len(raw.records)==len(rows)
    return dict(n=len(rows),valid=sum(r['valid'] for r in rows.values()),
                before=sum(r['pre'] for r in rows.values()),after=sum(r['post'] for r in rows.values()),
                gain=sum(r['post']>r['pre'] for r in rows.values()),loss=sum(r['post']<r['pre'] for r in rows.values()),
                qualified_gains=supported,memory_records=len(raw.records))


if __name__=='__main__':
    print(json.dumps(collect(*list(map(Path,sys.argv[1:3])),list(map(Path,sys.argv[3:]))),indent=2))
