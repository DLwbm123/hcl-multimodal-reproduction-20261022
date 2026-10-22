"""Offline checks for released data IDs, gates, matrices, costs and entrypoint syntax."""
import ast,json
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads((R/p).read_text())
native=read('results/sample_ids_native.json');pilot=read('results/sample_ids_pilot.json')
sets={s:{x['image_id'] for t in native['records'].values() for x in t[s]} for s in ['train','validation','test']}
assert not(sets['train']&sets['validation'] or sets['train']&sets['test'] or sets['validation']&sets['test'])
old={x['image_id'] for t in pilot['records'].values() for x in t['test']}
assert not sets['test']&old
for t in native['records'].values():
 for s,n in [('train',50),('validation',10),('test',100)]:assert len(t[s])==n
s=read('configs/native_stability_hcl.template.json');p=read('configs/native_plasticity_hcl.template.json')
a=s['optimizer'].copy();b=p['optimizer'].copy()
assert a.pop('historical_loss_budget')==0 and b.pop('historical_loss_budget') is None and a==b
assert a['min_format_compliance_rate']==1 and a['min_primary_score_delta']>0
for c in [s,p]:
 assert c['memory']['anchor_identity_mode']=='visible_input' and c['memory']['anchor_count_per_batch']==10
 assert not c['forgetting_probe']['enabled']
v=read('results/aggregate_results.json');total=0
for profile,arms in v['profiles'].items():
 for arm,x in arms.items():total+=x['cost']['calls']
assert total==19607
for arm,x in v['profiles']['native_fail_closed_v1'].items():
 assert x['exit']['exit_code']==0 and x['completed_stages']==4
 assert x['test_read_only']=={'checks':13,'all_passed':True}
 a=x['R'];assert abs(sum(a[-1])*25-x['final_average_100'])<1e-10
 f=sum(max(row[j] for row in a[j:])-a[-1][j] for j in range(3))*100/3
 assert abs(f-x['average_forgetting_pp'])<1e-10
 for stage in x['stage_metrics']:
  assert len(stage['tasks'])==stage['stage'] and all(t['total']==100 for t in stage['tasks'].values())
for name in ['r000.py','r001.py','r002.py']:ast.parse((R/name).read_text())
print('Release checks passed: IDs, leakage boundary, gates, read-only receipts, full denominators, metrics, costs and entrypoint syntax.')
