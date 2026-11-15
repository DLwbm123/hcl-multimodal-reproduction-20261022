"""Matched-method accuracy, mechanism ablations, history, and actual compute receipts."""
import json,random,math
from pathlib import Path

def rows(p):return [json.loads(l) for l in p.read_text().splitlines()]
def paired(a,b):
 assert len(a)==len(b)==100
 d=[int(x)-int(y) for x,y in zip(a,b)];g=d.count(1);l=d.count(-1);n=g+l;rng=random.Random(20261115)
 draws=sorted(sum(rng.choices(d,k=100)) for _ in range(20000));p=min(1.,2*sum(math.comb(n,k) for k in range(min(g,l)+1))/2**n) if n else 1.
 return {'gain':g,'loss':l,'delta_pp':sum(d),'paired95_pp':[draws[499],draws[19499]],'mcnemar_two_sided_p':p}
def analyze(root):
 R=Path(root);P=json.loads((R/'protocol.json').read_text());members=json.loads((R/'private_membership.json').read_text())['members'];expected={x['id'] for k,v in members.items() if k.startswith('confirmation:') for x in v};assert len(expected)==100
 train={x['id'] for k,v in members.items() if k.startswith('train:') for x in v};scores={};metrics={};graphs={};costs={};learning={}
 for arm in P['confirmation']['arms']:
  O=R/f'confirm_{arm}';res=rows(O/'private_scores.jsonl');assert len(res)==1
  result=res[0];scores[arm]={x['task_id']:x['success'] for x in result['case_results']};assert len(result['case_results'])==100 and set(scores[arm])==expected
  metrics[arm]=result['metrics'];graphs[arm]=json.loads((O/'GRAPH_SUMMARY.json').read_text());assert graphs[arm]['episodes']==100
  check=rows(O/'read_only_checks.jsonl')[-1];assert check['unchanged'] and check['weights_frozen']
  audit=rows(O/'private_retrieval_audit.jsonl');assert len(audit)==100 and {v['query_id'] for v in audit}==expected and all(set(v['source_ids'])<=train and v['query_id'] not in v['source_ids'] for v in audit)
  graphs[arm]['queries_with_reference']=sum(v['reference_count']>0 for v in audit);graphs[arm]['queries_with_tool_evidence']=sum(v['tool_evidence_present'] for v in audit)
  graphs[arm]['mean_tool_calls']=sum(graphs[arm]['tool_counts'].values())/100
  calls=rows(O/'call_costs.jsonl');f=O/'private_feature_costs.jsonl';features=rows(f) if f.exists() else []
  costs[arm]={'generation_calls':len(calls),'generation_seconds':sum(v['seconds'] for v in calls),'feature_forward_calls':len(features),'feature_seconds':sum(v['seconds'] for v in features),'elapsed_seconds':json.loads((O/'EXIT.json').read_text())['ended']-json.loads((O/'START.json').read_text())['started']}
 ids=sorted(expected);pairs={a:paired([scores['GN'][i] for i in ids],[scores[a][i] for i in ids]) for a in ['GR','G0','GP','GE','GF']}
 for role in ['fresh','reference']:
  O=R/role;gates=rows(O/f'state/{role}/evaluator/memory_update_history.jsonl');receipts=rows(O/'learning_receipts.jsonl');assert len(gates)==len(receipts)==2 and all(v['graph_training']['episodes']==24 for v in receipts)
  rs=rows(O/'private_scores.jsonl');s0={v['task_id']:v['success'] for v in next(v for v in rs if v['split']=='historical_initial')['case_results']};s1={v['task_id']:v['success'] for v in next(v for v in rs if v['split']=='historical_final')['case_results']};assert set(s0)==set(s1) and len(s0)==72
  learning[role]={'memory_accepts':sum(g['accepted'] for g in gates),'batches':receipts,'historical_retention':{'initial_correct':sum(s0.values()),'final_correct':sum(s1.values()),'losses':sum(s0[i] and not s1[i] for i in s0),'recoveries':sum(not s0[i] and s1[i] for i in s0),'total':72}}
 result={'confirmation':metrics,'primary_GN_vs_GR':pairs['GR'],'paired_GN_vs':pairs,'graph':graphs,'costs':costs,'learning':learning,'limitations':P['limitations'],'full_agent_learning_completed':True};(R/'ANALYSIS.json').write_text(json.dumps(result,indent=2))
 lines=['经验适用性与工具指导：两批匹配实验，独立100例确认']+[f"{a}: {metrics[a]['correct']}/100; completed {graphs[a]['completed']}/100; mean tools {graphs[a]['mean_tool_calls']:.2f}; gen {costs[a]['generation_calls']}; feature {costs[a]['feature_forward_calls']}" for a in P['confirmation']['arms']]
 lines+=[f'主比较GN-GR: {pairs["GR"]}','仅在新面板内比较，消融为次要比较；患者独立未知，不是官方test。'];(R/'INTERVIEW_REPORT.txt').write_text('\n'.join(lines)+'\n');return result
