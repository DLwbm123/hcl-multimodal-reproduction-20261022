"""Report both learning starts, all controls, fixed denominators and paired changes."""
import json,random,math
from pathlib import Path

def paired(a,b):
 assert len(a)==len(b)==200
 d=[int(x)-int(y) for x,y in zip(a,b)];gain=d.count(1);loss=d.count(-1);rng=random.Random(20261115)
 samples=sorted(sum(rng.choices(d,k=200))/200 for _ in range(20000));discord=gain+loss
 p=min(1.,2*sum(math.comb(discord,k) for k in range(min(gain,loss)+1))/2**discord) if discord else 1.
 return {'gain':gain,'loss':loss,'unchanged':200-gain-loss,'delta_pp':sum(d)/2,'paired95_pp':[100*samples[499],100*samples[19499]],'mcnemar_two_sided_p':p}
def analyze(root):
 R=Path(root);P=json.loads((R/'protocol.json').read_text());scores={};metrics={};graphs={};costs={}
 members=json.loads((R/'private_membership.json').read_text())['members'];expected={x['id'] for k,v in members.items() if k.startswith('confirmation:') for x in v};assert len(expected)==200
 for arm in P['confirmation']['arms']:
  out=R/f'confirm_{arm}';rs=[json.loads(x) for x in (out/'private_scores.jsonl').read_text().splitlines()];assert len(rs)==1
  result=rs[0];rows=result['case_results'];assert len(rows)==200;scores[arm]={x['task_id']:x['success'] for x in rows};assert set(scores[arm])==expected
  metrics[arm]=result['metrics'];graphs[arm]=json.loads((out/'GRAPH_SUMMARY.json').read_text());checks=[json.loads(x) for x in (out/'read_only_checks.jsonl').read_text().splitlines()];assert checks[-1]['unchanged'] and checks[-1]['weights_frozen']
  if arm.startswith('G'):assert graphs[arm]['episodes']==200
  calls=[json.loads(x) for x in (out/'call_costs.jsonl').read_text().splitlines()];costs[arm]={'calls':len(calls),'model_seconds':sum(x['seconds'] for x in calls),'seconds_per_case':sum(x['seconds'] for x in calls)/200}
 ids=sorted(expected);comparisons=[('GN','G0'),('GR','GT'),('GN','GF'),('GR','GF'),('GT','DT'),('GN','D0'),('GR','D0'),('GN','DT'),('GR','DT')]
 pairs={a+'_vs_'+b:paired([scores[a][i] for i in ids],[scores[b][i] for i in ids]) for a,b in comparisons}
 names=sorted(['GN_vs_G0','GR_vs_GT'],key=lambda k:pairs[k]['mcnemar_two_sided_p']);adjusted=0
 for rank,n in enumerate(names):
  adjusted=max(adjusted,min(1.,(2-rank)*pairs[n]['mcnemar_two_sided_p']));pairs[n]['holm_adjusted_p']=adjusted
 learning={}
 for role in ['fresh','resume']:
  out=R/role;events=[json.loads(x) for x in (out/f'state/{role}/evaluator/memory_update_history.jsonl').read_text().splitlines()];receipts=[json.loads(x) for x in (out/'learning_receipts.jsonl').read_text().splitlines()];assert len(events)==2 and len(receipts)==2
  assert all(x['graph_training']['episodes']==24 for x in receipts)
  h=[json.loads(x) for x in (out/'private_scores.jsonl').read_text().splitlines()];h0=next(x for x in h if x['split']=='historical_initial');h1=next(x for x in h if x['split']=='historical_final')
  s0={x['task_id']:x['success'] for x in h0['case_results']};s1={x['task_id']:x['success'] for x in h1['case_results']};assert set(s0)==set(s1) and len(s0)==72
  retention={'total':72,'initial_correct':sum(s0.values()),'final_correct':sum(s1.values()),'cumulative_losses':sum(s0[i] and not s1[i] for i in s0),'recoveries':sum(not s0[i] and s1[i] for i in s0)}
  learning[role]={'memory_gate_accepts':sum(bool(e.get('accepted')) for e in events),'memory_gate_total':len(events),'batches':receipts,'historical_retention':retention}
 report={'confirmation':metrics,'paired':pairs,'learning':learning,'graph':graphs,'costs':costs,'full_agent_learning_completed':True,'limitations':P['limitations']};(R/'ANALYSIS.json').write_text(json.dumps(report,indent=2))
 lines=['LangGraph 完整持续学习：最终确认结果','本报告来自预先冻结的7组×200例确认，不是官方test，患者独立性未知。','']
 for arm in P['confirmation']['arms']:lines.append(f"{arm}: {metrics[arm]['correct']}/200; Graph completion {graphs[arm].get('completed','not applicable')}")
 for role,pair in [('fresh','GN_vs_G0'),('resume','GR_vs_GT')]:
  q=pairs[pair];lines.append(f"{role}: 接受记忆更新 {learning[role]['memory_gate_accepts']}/2；学习净变化 {q['delta_pp']:+.1f}个百分点；gain {q['gain']} / loss {q['loss']}；95%区间 {q['paired95_pp']}；Holm校正p {q['holm_adjusted_p']:.4g}")
 lines+=['','必须同时披露固定6示例GF的结果；流程完成不能替代效果证据。','两学习起点和所有对照均保留，不按确认成绩选择种子、样本或checkpoint。']
 (R/'INTERVIEW_REPORT.txt').write_text('\n'.join(lines)+'\n');return report
