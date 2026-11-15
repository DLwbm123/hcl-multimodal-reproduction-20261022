"""Paired comparisons, fixed denominators and learning/graph evidence kept separate."""
import json,random,math
from pathlib import Path

def paired(a,b,n=200):
 assert len(a)==len(b)==n
 d=[int(x)-int(y) for x,y in zip(a,b)];gain=d.count(1);loss=d.count(-1);rng=random.Random(20261112)
 samples=sorted(sum(rng.choices(d,k=n))/n for _ in range(20000));discord=gain+loss
 p=min(1.,2*sum(math.comb(discord,k) for k in range(min(gain,loss)+1))/2**discord) if discord else 1.
 return {'gain':gain,'loss':loss,'unchanged':n-gain-loss,'delta_pp':100*sum(d)/n,'paired95_pp':[100*samples[499],100*samples[19499]],'mcnemar_two_sided_p':p}

def analyze(root):
 R=Path(root);P=json.loads((R/'protocol.json').read_text());scores={};metrics={};graphs={};costs={}
 members=json.loads((R/'private_membership.json').read_text())['members'];expected={x['id'] for k,v in members.items() if k.startswith('confirmation:') for x in v};assert len(expected)==200
 for arm in P['confirmation']['arms']:
  out=R/f'confirm_{arm}';rs=[json.loads(x) for x in (out/'private_scores.jsonl').read_text().splitlines()];assert len(rs)==1
  result=rs[0];rows=result['case_results'];assert len(rows)==200
  scores[arm]={x['task_id']:x['success'] for x in rows};assert set(scores[arm])==expected
  metrics[arm]=result['metrics'];graphs[arm]=json.loads((out/'GRAPH_SUMMARY.json').read_text())
  checks=[json.loads(x) for x in (out/'read_only_checks.jsonl').read_text().splitlines()];assert checks[-1]['unchanged'] and checks[-1]['weights_frozen']
  if arm.startswith('G'):assert graphs[arm]['episodes']==200
  calls=[json.loads(x) for x in (out/'call_costs.jsonl').read_text().splitlines()];costs[arm]={'calls':len(calls),'model_seconds':sum(x['seconds'] for x in calls),'seconds_per_case':sum(x['seconds'] for x in calls)/200}
 ids=sorted(expected);pairs={a+'_vs_'+b:paired([scores[a][i] for i in ids],[scores[b][i] for i in ids]) for a,b in [('GN','G0'),('GN','GF'),('GN','GT'),('GT','DT'),('GN','DT'),('GN','D0')]}
 out=R/'fresh';events=[json.loads(x) for x in (out/'state/fresh/evaluator/memory_update_history.jsonl').read_text().splitlines()];receipts=[json.loads(x) for x in (out/'learning_receipts.jsonl').read_text().splitlines()]
 assert len(events)==2 and len(receipts)==2
 assert all(x['graph_training']['episodes']==24 for x in receipts),'training bypassed Agent'
 h=[json.loads(x) for x in (out/'private_scores.jsonl').read_text().splitlines()];h0=next(x for x in h if x['split']=='historical_initial');h1=next(x for x in h if x['split']=='historical_final')
 s0={x['task_id']:x['success'] for x in h0['case_results']};s1={x['task_id']:x['success'] for x in h1['case_results']};assert set(s0)==set(s1) and len(s0)==72
 retention={'total':72,'initial_correct':sum(s0.values()),'final_correct':sum(s1.values()),'cumulative_losses':sum(s0[i] and not s1[i] for i in s0),'recoveries':sum(not s0[i] and s1[i] for i in s0)}
 learning={'memory_gate_accepts':sum(bool(e.get('accepted')) for e in events),'memory_gate_total':len(events),'batches':receipts,'historical_retention':retention}
 main=pairs['GN_vs_G0'];report={'confirmation':metrics,'paired':pairs,'learning':learning,'graph':graphs,'costs':costs,'full_agent_learning_completed':True,'positive_vs_initial_agent':main['delta_pp']>0,'paired95_lower_positive':main['paired95_pp'][0]>0,'beats_fixed_examples':pairs['GN_vs_GF']['delta_pp']>0,'limitations':P['limitations']}
 (R/'ANALYSIS.json').write_text(json.dumps(report,indent=2));return report
