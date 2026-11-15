"""Training-only empirical arbitration between guided and independent observations."""
import json,random,math
from pathlib import Path
from collections import defaultdict
from adaptive import intent
from contracts import rows,score_map

def use_independent(question,policy):
 return bool(policy and intent(question) in policy.get('allowed_intents',[]))

def fit(root):
 root=Path(root);p=json.loads((root/'protocol.json').read_text());train=json.loads((root/'private_learning.json').read_text())['train']
 assert len(train)==48 and all(e['split']=='train' for e in train)
 a,_=score_map(root/'fit_base','counterfactual_train');b,_=score_map(root/'fit_evidence','counterfactual_train');assert set(a)==set(b)=={e['task_id'] for e in train}
 groups=defaultdict(lambda:{'support':0,'base_correct':0,'evidence_correct':0,'gains':0,'losses':0})
 for e in train:
  k=intent(e['question']);sid=e['task_id'];g=groups[k];g['support']+=1;g['base_correct']+=a[sid];g['evidence_correct']+=b[sid];g['gains']+=int(b[sid] and not a[sid]);g['losses']+=int(a[sid] and not b[sid])
 rule=p['routing'];allowed=sorted(k for k,g in groups.items() if g['support']>=rule['min_support'] and g['gains']-g['losses']>=rule['min_net_training_gain'] and g['losses']<=rule['max_training_losses'])
 result={'allowed_intents':allowed,'training_groups':dict(groups),'training_count':48,'selection_used':False,'confirmation_used':False,'fallback':'base','rule':{k:rule[k] for k in ['min_support','min_net_training_gain','max_training_losses']}}
 (root/'ROUTING_CANDIDATE.json').write_text(json.dumps(result,indent=2));return result

def gate(root):
 root=Path(root);a,_=score_map(root/'eval_base','selection');b,m=score_map(root/'eval_routed','selection');h0,_=score_map(root/'eval_base','historical');h1,_=score_map(root/'eval_routed','historical');assert len(a)==48 and set(a)==set(b) and len(h0)==72 and set(h0)==set(h1)
 gain=sum(b.values())-sum(a.values());loss=sum(h0[k] and not h1[k] for k in h0);reasons=[]
 if gain<1:reasons.append('current_correct_gain_below_one')
 if loss>2:reasons.append('historical_loss_budget_exceeded')
 if m['format_compliance_rate']<.95:reasons.append('format_compliance_below_95pct')
 result={'accepted':not reasons,'reasons':reasons,'selection_before':sum(a.values()),'selection_after':sum(b.values()),'selection_total':48,'current_correct_gain':gain,'paired_gains':sum(b[k] and not a[k] for k in a),'paired_losses':sum(a[k] and not b[k] for k in a),'selection_format':m['format_compliance_rate'],'historical_before':sum(h0.values()),'historical_after':sum(h1.values()),'historical_losses':loss,'historical_recoveries':sum(h1[k] and not h0[k] for k in h0),'historical_total':72}
 (root/'ROUTING_GATE.json').write_text(json.dumps(result,indent=2));return result

def analyze(root):
 root=Path(root);policy=json.loads((root/'ROUTING_CANDIDATE.json').read_text());g=json.loads((root/'ROUTING_GATE.json').read_text());out={'policy':policy,'gate':g,'confirmation_opened':False,'roles':{}}
 roles=['fit_base','fit_evidence']+(['eval_base','eval_evidence','eval_routed'] if policy['allowed_intents'] else [])+(['confirm_base','confirm_routed','confirm_evidence'] if g['accepted'] else [])
 for role in roles:
  d=root/role;assert json.loads((d/'EXIT.json').read_text())['exit_code']==0;calls=rows(d/'call_costs.jsonl');assert len(calls)==json.loads((d/'model_attempts.json').read_text())['attempts'];features=rows(d/'private_feature_costs.jsonl');traces=rows(d/'private_graph_traces.jsonl');audit=rows(d/'private_retrieval_audit.jsonl');scores=rows(d/'private_scores.jsonl');ro=rows(d/'read_only_checks.jsonl');assert len(ro)==len(scores) and all(x['unchanged'] and x['weights_frozen'] for x in ro)
  out['roles'][role]={'metrics':{x['split']:x['metrics'] for x in scores},'generation_calls':len(calls),'generation_failed':sum(not x['success'] for x in calls),'generation_seconds':sum(x['seconds'] for x in calls),'feature_forwards':len(features),'feature_seconds':sum(x['seconds'] for x in features),'graph_completed':sum(x['status']=='completed' for x in traces),'graph_failed':sum(x['status']!='completed' for x in traces),'tool_calls':sum(ev['node']=='execute' for x in traces for ev in x['events']),'independent_selected':sum(x['independent_selected'] for x in audit),'read_only_and_weights_frozen':True}
 if g['accepted']:
  p=json.loads((root/'protocol.json').read_text());a,_=score_map(root/'confirm_routed','confirmation');ids={x['task_id'] for x in json.loads((root/'private_confirmation.json').read_text())};assert len(ids)==100 and set(a)==ids;out['paired_routed_vs']={}
  for role in ['base','evidence']:
   b,_=score_map(root/('confirm_'+role),'confirmation');assert set(b)==ids;diff=[int(a[k])-int(b[k]) for k in sorted(ids)];gain=diff.count(1);loss=diff.count(-1);n=gain+loss;rng=random.Random(p['seed']);boot=sorted(sum(rng.choices(diff,k=100)) for _ in range(20000));out['paired_routed_vs'][role]={'gain':gain,'loss':loss,'delta_pp':sum(diff),'paired95_pp':[boot[499],boot[19499]],'mcnemar_p':min(1.,2*sum(math.comb(n,k) for k in range(min(gain,loss)+1))/2**n) if n else 1.}
  out['confirmation_opened']=True
 (root/'ANALYSIS.json').write_text(json.dumps(out,indent=2));return out
