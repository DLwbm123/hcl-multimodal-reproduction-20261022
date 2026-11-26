"""Release only aggregate patient and action diagnostics from private trajectories."""
from collections import Counter
import json
from pathlib import Path
def read(name):return json.loads(Path(name+'.json').read_text())
cs={c['id']:c for c in read('cases')}
stats={}
for phase,arms in [('build',('plain','demo')),('test',('plain','demo','score','credit'))]:
    for arm in arms:
        rows=[r for i in range(4) for r in read(f'{phase}_{arm}_{i}')]
        s=Counter();by_person=Counter()
        for row in rows:
            case=cs[row['id']];h=row['h']
            by_person[case['person']]+=row['hit']
            call=h[0][1];reply=h[1][1] if len(h)>1 else ''
            tag='read' if call.startswith('READ ') and reply.startswith('{"handle":') else 'finish' if call.startswith('FINISH ') and len(h)==1 else 'invalid'
            s['first_'+tag]+=1
            s['numeric_success' if case['truth'] is not None else 'missing_success']+=row['hit']
            s['evidence_'+row['cause']]+=1
        s['patients']=len(by_person);s['patients_any_success']=sum(x>0 for x in by_person.values())
        stats[phase+'_'+arm]=s
print(json.dumps(stats,indent=2))
