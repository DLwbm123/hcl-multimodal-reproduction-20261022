import os
import sys,json,collections
from pathlib import Path
import numpy as np
R=Path(sys.argv[1]);sys.path.insert(0,os.environ['MEDICAL_SOURCE'])
from medical_hcl.evaluation import score_case
rows=lambda n:[json.loads(x) for x in (R/n).read_text().splitlines()]
preds=rows('private_predictions.jsonl');checks=rows('read_only_checks.jsonl');costs=rows('call_costs.jsonl')
assert len(preds)==len(checks)==486 and len(costs)==1794
assert [x['attempt'] for x in costs]==list(range(1,1795))
assert all(x['learning_state_unchanged'] and x['parameters_frozen'] for x in checks)
assert sum(x['retrieval_matched'] for x in checks)==186
examples={x['task_id']:x for x in rows('private_confirmation.jsonl')};vocab=json.loads((R/'class_vocabulary.json').read_text())
assert len(examples)==150 and len({x['metadata']['group_id'] for x in examples.values()})==150
conditions=['static_no_memory','raw_only','raw_semantic'];tasks=['classification','polyp_localization','visual_qa'];scores={};usage={};phases={}
for c in conditions:
 ps=[x for x in preds if x['stage']=='confirmation' and x['condition']==c]
 assert len(ps)==150 and {x['prediction']['task_id'] for x in ps}==set(examples)
 scores[c]={};u=collections.Counter()
 for p in ps:
  p=p['prediction'];e=examples[p['task_id']];valid,success,score,aux=score_case(e,p['answer'],vocab)
  scores[c][p['task_id']]=int(success)
  mem=p.get('trace',{}).get('selected_memory',[]);u['examples_with_memory']+=bool(mem);u['selected_records']+=len(mem)
 usage[c]=dict(u)
 phases[c]=dict(collections.Counter(x['phase'] for x in costs if x['condition']==c))
rng=np.random.default_rng(20261023);paired={}
for left,right in [('static_no_memory','raw_only'),('static_no_memory','raw_semantic'),('raw_only','raw_semantic')]:
 pair={}
 for t in tasks+['all']:
  ids=sorted(k for k,e in examples.items() if t=='all' or e['task_name']==t)
  a=np.array([scores[left][k] for k in ids]);b=np.array([scores[right][k] for k in ids]);d=b-a
  bs=d[rng.integers(0,len(d),(20000,len(d)))].mean(axis=1)*100
  pair[t]={'n':len(ids),'left_correct':int(a.sum()),'right_correct':int(b.sum()),'success_to_failure':int(((a==1)&(b==0)).sum()),'failure_to_success':int(((a==0)&(b==1)).sum()),'delta_pp':float(d.mean()*100),'paired_bootstrap_95_percentile_pp':np.percentile(bs,[2.5,97.5]).tolist()}
 paired[right+'_vs_'+left]=pair
out={'new_predictions':486,'reused_screen_predictions':72,'readonly_checks_passed':486,'semantic_invariance_checks_passed':186,'confirmation_groups':150,'all_conditions_same_ids':True,'test_read':False,'calls':1794,'call_seconds':sum(x['seconds'] for x in costs),'calls_by_condition_phase':phases,'confirmation_memory_use':usage,'paired_confirmation':paired,'bootstrap':{'seed':20261023,'resamples':20000,'unit':'content group (one case each); pooled all is descriptive','interval':'percentile paired bootstrap; small samples, not multiplicity adjusted'},'exit':json.loads((R/'EXIT.json').read_text())}
(R/'completion_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
