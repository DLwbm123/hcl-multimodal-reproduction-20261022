import os
import json,sys,math
from pathlib import Path
import numpy as np
R=Path(sys.argv[1]);sys.path.insert(0,os.environ['MEDICAL_SOURCE'])
from medical_hcl.evaluation import score_case
rows=lambda p:[json.loads(x) for x in p.read_text().splitlines()]
es={e['task_id']:e for e in rows(R/'private_confirmation.jsonl')};assert len(es)==200 and len({x['metadata']['group_id'] for x in es.values()})==200
vocab=json.loads((R/'class_vocabulary.json').read_text());out={'arms':{},'test_used':False,'learning_performed':False};scores={}
for folder,arm in [('baselines','static_no_memory'),('hcl_scoped','hcl_scoped')]:
 O=R/folder;end=json.loads((O/'EXIT.json').read_text());assert end['exit_code']==0
 ps=rows(O/'private_predictions.jsonl');cs=rows(O/'read_only_checks.jsonl');calls=rows(O/'call_costs.jsonl')
 assert len(ps)==len(cs)==200 and {p['prediction']['task_id'] for p in ps}==set(es)
 assert all(c['learning_state_unchanged'] and c['parameters_frozen'] for c in cs)
 assert len(calls)==end['calls'] and [c['attempt'] for c in calls]==list(range(1,len(calls)+1))
 scores[arm]={p['prediction']['task_id']:score_case(es[p['prediction']['task_id']],p['prediction']['answer'],vocab) for p in ps}
 out['arms'][arm]={'correct':sum(int(v[1]) for v in scores[arm].values()),'format':sum(int(v[0]) for v in scores[arm].values()),'total':200,'calls':len(calls),'call_seconds':sum(x['seconds'] for x in calls),'exit':end,'selected_memory_examples':sum(bool(p['prediction'].get('trace',{}).get('selected_memory')) for p in ps)}
ids=sorted(es);a=np.array([scores['static_no_memory'][k][1] for k in ids],dtype=int);b=np.array([scores['hcl_scoped'][k][1] for k in ids],dtype=int);d=b-a;lost=int(((a==1)&(b==0)).sum());gain=int(((a==0)&(b==1)).sum());n=lost+gain
rng=np.random.default_rng(2026102304);bs=d[rng.integers(0,200,(20000,200))].mean(axis=1)*100;ci=np.percentile(bs,[2.5,97.5]).tolist()
out['paired']={'lost':lost,'recovered':gain,'delta_pp':float(d.mean()*100),'paired_bootstrap_95_pp':ci,'mcnemar_exact_two_sided_p':min(1.,2*sum(math.comb(n,k) for k in range(min(lost,gain)+1))/2**n) if n else 1.}
out['gate']={'gain_at_least_5pp':int(b.sum()-a.sum())>=10,'paired_ci_lower_positive':ci[0]>0};out['gate']['passed']=all(out['gate'].values())
ss=rows(R/'hcl_scoped/task_scope_checks.jsonl');assert all(x['all_candidates_source_task_matched'] for x in ss);out['scope_checks']=len(ss)
(R/'completion_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
