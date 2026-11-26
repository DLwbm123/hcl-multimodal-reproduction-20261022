"""Score anonymous follow-up judgments against the frozen main-run table."""
import json
import sys
from pathlib import Path


def read(root, name):
    return json.loads((root/(name+'.json')).read_text())


def summarize(root, main_root):
    grades=read(root,'private_grades')
    blind={r['item']:r['answers'] for r in read(root,'private_blind')}
    table={r['item']:r for r in read(main_root,'paired_table')}
    rows=[]
    for r in read(root,'private_control').values():
        i=r['item']
        labels=grades[str(i)]
        assert len(blind[i])==len(labels) and set(labels)<=set('01')
        score=dict(zip(blind[i],labels))
        rows.append((score[r['retained']],score[r['global_answer']],table[i]['V']))
    assert len(rows)==len(blind)==len(grades)==7
    tokens=[json.loads(x) for x in (root/'tokens.jsonl').read_text().splitlines()]
    return dict(n=len(rows), arms=['reuse_old','whole','visual'],
                scores=[sum(int(r[j]) for r in rows) for j in range(3)],
                global_only_vs_local=sum(r[1:]==('1','0') for r in rows),
                local_only_vs_global=sum(r[1:]==('0','1') for r in rows),
                both=sum(r[1:]==('1','1') for r in rows),runtime=read(root,'runtime'),
                tokens_in=sum(x[1] for x in tokens), tokens_out=sum(x[2] for x in tokens))


if __name__=='__main__':
    print(json.dumps(summarize(*map(Path,sys.argv[1:])),indent=2))
