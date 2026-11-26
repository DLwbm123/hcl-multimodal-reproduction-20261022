import json,os
from pathlib import Path
from clinical import split
from feedback import outcome,future
assert outcome(0,1)=='corrected' and outcome(1,0)=='spoiled'
assert future(0,[])=='unobserved' and future(0,[0,1])=='later_corrected'
fit,gate,test=split()
assert (len(fit),len(gate),len(test))==(30,16,45)
src=Path(os.environ['PAIRED_RUN'])
assert json.loads((src/'SUPERVISOR_EXIT.json').read_text())['success']
for phase,keys,n in [('gate',gate,1),('test',test,8)]:
    rows=json.loads((src/('private_'+phase+'.json')).read_text())
    assert set(rows)==set(keys) and sum(r['base_score']['correct'] for r in rows.values())==n
print(json.dumps({'passed':True}))
