import json,collections
from pathlib import Path
import numpy as np
from evaluate import aggregate

def analyze(root):
 root=Path(root);out=root/'run';p=json.loads((root/'protocol.json').read_text());r=json.loads((out/'private_confirmation_results.json').read_text());learning=json.loads((out/'learning_summary.json').read_text());rng=np.random.default_rng(p['seed'])
 summary={arm:{task:aggregate(rows) for task,rows in tasks.items()} for arm,tasks in r.items()}
 def paired(a,b,task=None):
  tasks=[task] if task else p['tasks'];arrays=[]
  for t in tasks:
   assert [x['group_id'] for x in r[a][t]]==[x['group_id'] for x in r[b][t]]
   arrays.append([x['metrics']['score']-y['metrics']['score'] for x,y in zip(r[a][t],r[b][t])])
  delta=np.mean(arrays,axis=0);n=len(delta);bs=delta[rng.integers(0,n,size=(p['evaluation']['bootstrap_samples'],n))].mean(1)
  return {'source_groups':n,'mean_difference':float(delta.mean()),'paired_group_bootstrap95':np.quantile(bs,[.025,.975]).tolist(),'positive_groups':int((delta>0).sum()),'negative_groups':int((delta<0).sum())}
 comparisons={'S2_vs_RAW_macro':paired('S2','RAW'),'S2_vs_FIXED_macro':paired('S2','FIXED'),'T1_S1_vs_S0':paired('S1','S0','T1'),'T1_S2_vs_S1':paired('S2','S1','T1'),'T2_S2_vs_S1':paired('S2','S1','T2')}
 process={}
 for task in p['tasks']:
  a=r['S2'][task];b=r['RAW'][task];seq_changes=arg_changes=0;tools=collections.Counter();branches=collections.Counter();uses=collections.Counter()
  for x,y in zip(a,b):
   xe=x['episode'];ye=y['episode'];seq_changes += [t['tool'] for t in xe['tools']]!=[t['tool'] for t in ye['tools']]
   arg_changes += [(t['tool'],t['args']) for t in xe['tools']]!=[(t['tool'],t['args']) for t in ye['tools']]
   tools.update(t['tool'] for t in xe['tools']);uses.update(xe['skills_used']);branches.update(str(z['executed']) for z in xe['branches'])
  process[task]={'skill_invocations':dict(uses),'primitive_tool_counts':dict(tools),'conditional_branches':dict(branches),'different_tool_sequence_vs_RAW':seq_changes,'different_calls_or_arguments_vs_RAW':arg_changes,'old_correct_lost_T1':sum(x['metrics']['success'] and not y['metrics']['success'] for x,y in zip(r['S1']['T1'],r['S2']['T1'])) if task=='T1' else None}
 invocations=sum(sum(x['skill_invocations'].values()) for x in process.values());primary=comparisons['S2_vs_RAW_macro'];accepted=sum(g['accepted'] for g in learning['gates'])
 no_observed_loss=all(summary['S2'][t]['mean_score']>=summary['RAW'][t]['mean_score']-1e-12 and summary['S2'][t]['correct']>=summary['RAW'][t]['correct'] for t in p['tasks'])
 means=lambda arm,key:float(np.mean([summary[arm][t][key] for t in p['tasks']]))
 efficiency={'no_observed_task_score_or_success_loss':no_observed_loss,'primitive_call_reduction':means('RAW','mean_primitive_calls')-means('S2','mean_primitive_calls'),'vlm_call_reduction_fraction':1-means('S2','mean_vlm_calls')/max(1e-9,means('RAW','mean_vlm_calls'))}
 efficiency['descriptive_signal']=no_observed_loss and efficiency['primitive_call_reduction']>=1 and efficiency['vlm_call_reduction_fraction']>=.2
 verdict=('No candidate skill passed the frozen validation gate.' if not accepted else 'Skills were retained but none were invoked on confirmation.' if not invocations else 'Exploratory positive skill accuracy increment; uncertainty and small-group limitations apply.' if primary['mean_difference']>0 else 'Descriptive efficiency signal without observed task-score loss; not proof of statistical non-inferiority.' if efficiency['descriptive_signal'] else 'Retained skills were invoked, but confirmation does not show an accuracy or predefined efficiency increment over matched raw memory.')
 report={'study':p['study'],'counts':json.loads((root/'DATA_CHECK.json').read_text()),'summary':summary,'comparisons':comparisons,'workflow':process,'efficiency_diagnostic':efficiency,'learning':learning,'verdict':verdict,'limitations':p['limitations'],'primary_source_groups':8,'confirmation_read_after_state_freeze':True}
 (root/'ANALYSIS.json').write_text(json.dumps(report,indent=2,allow_nan=False))
 (root/'RESULT.txt').write_text(verdict+'\n\n'+json.dumps({'summary':summary,'comparisons':comparisons,'workflow':process},indent=2)+'\n\nLimitations:\n'+'\n'.join(p['limitations']))
 return report
