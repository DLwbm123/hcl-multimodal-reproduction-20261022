"""Predeclared paired exact-match analysis; all 200 frozen groups retained."""
import json,random,math
from pathlib import Path

def paired(a,b):
 assert len(a)==len(b)==200
 d=[int(x)-int(y) for x,y in zip(a,b)];gain=d.count(1);loss=d.count(-1);rng=random.Random(20261110)
 samples=sorted(sum(rng.choices(d,k=len(d)))/len(d) for _ in range(20000));discord=gain+loss
 p=min(1.,2*sum(math.comb(discord,k) for k in range(min(gain,loss)+1))/2**discord) if discord else 1.
 return {'gain':gain,'loss':loss,'unchanged':len(d)-gain-loss,'delta_pp':100*sum(d)/len(d),'paired95_pp':[100*samples[499],100*samples[19499]],'mcnemar_two_sided_p':p}

def analyze(root):
 R=Path(root);frozen=json.loads((R/'FROZEN_SELECTION.json').read_text());scores={};metrics={}
 for arm in ['B','F','H','N']:
  rs=[json.loads(x) for x in (R/f'confirm_{arm}'/'private_scores.jsonl').read_text().splitlines()];assert len(rs)==1
  result=rs[0];rows=result['case_results'];assert len(rows)==200
  scores[arm]={x['task_id']:x['success'] for x in rows};metrics[arm]=result['metrics']
 ids=sorted(scores['N']);assert all(set(s)==set(ids) for s in scores.values())
 pairs={arm:paired([scores['N'][i] for i in ids],[scores[arm][i] for i in ids]) for arm in ['B','F','H']}
 learning={}
 for role in ['fresh','resume']:
  events=[json.loads(x) for x in (R/role/'state'/role/'evaluator/memory_update_history.jsonl').read_text().splitlines()]
  receipts=[json.loads(x) for x in (R/role/'learning_receipts.jsonl').read_text().splitlines()]
  learning[role]={'memory_gate_accepts':sum(bool(e.get('accepted')) for e in events),'memory_gate_total':len(events),'batches':receipts}
 main=pairs[frozen['primary_control']]
 result={'selected_arm':frozen['selected_arm'],'primary_control':frozen['primary_control'],'confirmation':metrics,'paired_N_vs':pairs,'interview_goal':{'full_loop_completed':True,'any_memory_update_accepted':any(x['memory_gate_accepts']>0 for x in learning.values()),'descriptive_gain_vs_primary_control':main['delta_pp']>0,'descriptive_gain_vs_old_state':pairs['H']['delta_pp']>0,'effectiveness_not_required_for_workflow_demo':True},'research_threshold_reference_only':main['delta_pp']>=5 and main['paired95_pp'][0]>0 and pairs['H']['delta_pp']>0,'learning':learning,'limits':['not original test','source/pixel groups independent; patients unverified','same model and one deterministic decoding seed','exact-match benefit does not establish clinical correctness']}
 (R/'ANALYSIS.json').write_text(json.dumps(result,indent=2));return result

if __name__=='__main__':
 import os
 analyze(os.environ['RUN_ROOT'])
