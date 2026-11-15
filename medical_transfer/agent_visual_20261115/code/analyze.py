import json,random,math
from pathlib import Path

def paired(a,b):
 assert len(a)==len(b)==100
 d=[int(x)-int(y) for x,y in zip(a,b)];gain=d.count(1);loss=d.count(-1);rng=random.Random(20261115)
 draws=sorted(sum(rng.choices(d,k=100)) for _ in range(20000));n=gain+loss;p=min(1.,2*sum(math.comb(n,k) for k in range(min(gain,loss)+1))/2**n) if n else 1.
 return {'gain':gain,'loss':loss,'unchanged':100-gain-loss,'delta_pp':sum(d),'paired95_pp':[draws[499],draws[19499]],'mcnemar_two_sided_p':p}
def analyze(root):
 R=Path(root);P=json.loads((R/'protocol.json').read_text());members=json.loads((R/'private_membership.json').read_text())['members'];expected={x['id'] for k,v in members.items() if k.startswith('confirmation:') for x in v};assert len(expected)==100
 scores={};metrics={};graphs={};costs={}
 for arm in ['G0','GN','GX','GF']:
  O=R/f'confirm_{arm}';rows=[json.loads(l) for l in (O/'private_scores.jsonl').read_text().splitlines()];assert len(rows)==1;result=rows[0]
  scores[arm]={x['task_id']:x['success'] for x in result['case_results']};assert len(result['case_results'])==100 and set(scores[arm])==expected
  metrics[arm]=result['metrics'];graphs[arm]=json.loads((O/'GRAPH_SUMMARY.json').read_text());assert graphs[arm]['episodes']==100
  checks=[json.loads(l) for l in (O/'read_only_checks.jsonl').read_text().splitlines()];assert checks[-1]['unchanged'] and checks[-1]['weights_frozen']
  calls=[json.loads(l) for l in (O/'call_costs.jsonl').read_text().splitlines()];costs[arm]={'calls':len(calls),'model_seconds':sum(x['seconds'] for x in calls)}
  refs=[json.loads(l) for l in (O/'private_retrieval_audit.jsonl').read_text().splitlines()];assert len(refs)==100
  graphs[arm]['queries_with_reference']=sum(x['reference_count']>0 for x in refs)
  graphs[arm]['reference_examples_total']=sum(x['reference_count'] for x in refs)
  graphs[arm]['visual_references_enabled']=arm!='GX'
 ids=sorted(expected);pairs={b:paired([scores['GN'][i] for i in ids],[scores[b][i] for i in ids]) for b in ['G0','GX','GF']}
 O=R/'fresh';gates=[json.loads(l) for l in (O/'state/fresh/evaluator/memory_update_history.jsonl').read_text().splitlines()];receipts=[json.loads(l) for l in (O/'learning_receipts.jsonl').read_text().splitlines()];assert len(gates)==1 and len(receipts)==1 and receipts[0]['graph_training']['episodes']==24
 rows=[json.loads(l) for l in (O/'private_scores.jsonl').read_text().splitlines()];s0={x['task_id']:x['success'] for x in next(r for r in rows if r['split']=='historical_initial')['case_results']};s1={x['task_id']:x['success'] for x in next(r for r in rows if r['split']=='historical_final')['case_results']};assert set(s0)==set(s1) and len(s0)==72
 history={'total':72,'initial_correct':sum(s0.values()),'final_correct':sum(s1.values()),'losses':sum(s0[i] and not s1[i] for i in s0),'recoveries':sum(not s0[i] and s1[i] for i in s0)}
 result={'confirmation':metrics,'paired_GN_vs':pairs,'graph':graphs,'costs':costs,'memory_accepted':bool(gates[0]['accepted']),'learning_receipts':receipts,'historical_retention':history,'limitations':P['limitations'],'full_single_batch_learning_completed':True};(R/'ANALYSIS.json').write_text(json.dumps(result,indent=2))
 lines=['LangGraph单批完整学习：100例视觉经验机制确认（与其他实验分开报告）']+[f"{a}: {metrics[a]['correct']}/100, 流程完成 {graphs[a]['completed']}/100" for a in ['G0','GN','GX','GF']]
 lines += [f"记忆更新接受: {gates[0]['accepted']}",f"GN-G0: {pairs['G0']}",f"GN-GF: {pairs['GF']}",f"历史保持: {history}",'这是单批小规模验证，不是完整多批研究结论或官方测试集。']
 (R/'INTERVIEW_REPORT.txt').write_text('\n'.join(lines)+'\n');return result
