import os
import sys,json,collections,math
from pathlib import Path
import numpy as np
R=Path(sys.argv[1]);sys.path.insert(0,os.environ['MEDICAL_SOURCE'])
from medical_hcl.evaluation import score_case,stage_summary
rows=lambda p:[json.loads(x) for x in p.read_text().splitlines()]
es={x['task_id']:x for x in rows(R/'private_confirmation.jsonl')};tasks=['classification','polyp_localization','visual_qa'];vocab=json.loads((R/'class_vocabulary.json').read_text());out={'arms':{},'integrity':{},'paired':{}};scores={}
assert len(es)==150 and len({e['metadata']['group_id'] for e in es.values()})==150
for dirname in ['baselines','hcl_contract','hcl_semantic']:
 o=R/dirname;exit=json.loads((o/'EXIT.json').read_text());assert exit['exit_code']==0
 ps=rows(o/'private_predictions.jsonl');checks=rows(o/'read_only_checks.jsonl');calls=rows(o/'call_costs.jsonl');assert len(ps)==len(checks)==300 and len(calls)==exit['calls']
 assert [x['attempt'] for x in calls]==list(range(1,len(calls)+1)) and all(x['learning_state_unchanged'] and x['parameters_frozen'] for x in checks)
 q=json.loads((o/'RESULT.json').read_text());out['integrity'][dirname]={'readonly_checks':len(checks),'calls':len(calls),'seconds':sum(x['seconds'] for x in calls),'calls_by_phase':dict(collections.Counter(x['phase'] for x in calls)),'exit':exit}
 for arm in (['direct','static_no_memory'] if dirname=='baselines' else [dirname]):
  final=[x for x in ps if x['arm']==arm and x['stage']==(0 if dirname=='baselines' else 3)];assert len(final)==150 and {x['prediction']['task_id'] for x in final}==set(es)
  metric={};scores[arm]={}
  for t in tasks:
   group=[x for x in final if x['task']==t];assert len(group)==50;vals=[score_case(es[x['prediction']['task_id']],x['prediction']['answer'],vocab) for x in group]
   metric[t]={'correct':sum(int(v[1]) for v in vals),'total':50,'format':sum(int(v[0]) for v in vals),'selected_memory_examples':sum(bool(x['prediction'].get('trace',{}).get('selected_memory')) for x in group)}
   for x,v in zip(group,vals):scores[arm][x['prediction']['task_id']]=int(v[1])
  result=q[arm] if dirname=='baselines' else q
  out['arms'][arm]={'metrics':metric,'matrix':result['matrix'],'summary':stage_summary(result['matrix'])}
  if dirname!='baselines':
   for stage in range(1,4):
    for t in tasks[:stage]:assert len([x for x in ps if x['stage']==stage and x['task']==t])==50
   history=result['optimizer_decisions'];rejs=[y for x in history for y in x.get('candidate_rejections',[])];mem=json.loads((o/'state'/arm/'evaluator/memory_update_history.json').read_text())
   out['arms'][arm].update(memory=result['memory'],accepted_prompts=[{k:x.get(k) for k in ['component','candidate_id','current_validation_primary_score','candidate_validation_primary_score','correct_gain','anchor_forget_count']} for x in history if x.get('accepted')],prompt_decision_count=len(history),candidate_rejection_reasons=dict(collections.Counter(reason for x in rejs for reason in x['reasons'])),candidate_records=len(rejs),memory_gates=[{'task':t,'accepted':m['accepted'],'reasons':m.get('rejection_reasons',[]),'stable_correct':m['stable_current_task_metrics']['correct'],'candidate_correct':m['candidate_current_task_metrics']['correct'],'candidate_format':m['candidate_current_task_metrics']['format_compliance_rate'],'anchor_total':m['candidate_historical_anchor_metrics']['total'],'anchor_lost':m['candidate_historical_anchor_metrics']['forget_count'],'anchor_recovered':m['candidate_historical_anchor_metrics']['recovered_count']} for t,m in zip(tasks,mem)])
rng=np.random.default_rng(20261023)
for ref,cur in [('direct','static_no_memory'),('static_no_memory','hcl_contract'),('static_no_memory','hcl_semantic'),('hcl_contract','hcl_semantic')]:
 pair={}
 for t in tasks+['all']:
  ids=sorted(k for k,e in es.items() if t=='all' or e['task_name']==t);a=np.array([scores[ref][k] for k in ids]);b=np.array([scores[cur][k] for k in ids]);d=b-a;loss=int(((a==1)&(b==0)).sum());gain=int(((a==0)&(b==1)).sum());n=loss+gain
  bs=d[rng.integers(0,len(d),(20000,len(d)))].mean(axis=1)*100
  pair[t]={'reference_correct':int(a.sum()),'current_correct':int(b.sum()),'n':len(d),'lost':loss,'recovered':gain,'delta_pp':float(d.mean()*100),'paired_bootstrap_95_pp':np.percentile(bs,[2.5,97.5]).tolist(),'mcnemar_exact_two_sided_p':min(1.,2*sum(math.comb(n,k) for k in range(min(loss,gain)+1))/2**n) if n else 1.}
 out['paired'][cur+'_vs_'+ref]=pair
m=out['arms'];out['gate']={'vqa_gain_at_least_5pp':m['hcl_semantic']['metrics']['visual_qa']['correct']-m['static_no_memory']['metrics']['visual_qa']['correct']>=3,'no_task_accuracy_or_format_degradation':all(m['hcl_semantic']['metrics'][t][k]>=m[a]['metrics'][t][k] for t in tasks for k in ['correct','format'] for a in ['static_no_memory','hcl_contract']),'pooled_interval_lower_above_zero':out['paired']['hcl_semantic_vs_static_no_memory']['all']['paired_bootstrap_95_pp'][0]>0}
out['gate']['passed']=all(out['gate'].values());out['total_calls']=sum(v['calls'] for v in out['integrity'].values());out['total_readonly_checks']=900;out['test_used']=False
(R/'completion_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
