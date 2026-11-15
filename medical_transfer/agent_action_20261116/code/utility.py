"""A training-only empirical action-value policy with an independent commit gate."""
import json,re,random,math
from collections import defaultdict
from pathlib import Path
from adaptive import intent

def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]
def observation_key(question,tool,result):
 if tool not in {'answer_visual_question','classify_image'} or intent(question)=='other':return None
 text=' '.join(str(result).lower().split()).strip(' .')
 if not text or len(text)>160 or re.search(r'\b(uncertain|unclear|cannot|unable|maybe|possibly|unknown|unsure)\b',text):return None
 if re.fullmatch(r'\d+',text):shape='digits'
 elif text in {'yes','no'}:shape='binary'
 elif len(text.split())<=6 and '\n' not in str(result):shape='short_text'
 else:return None
 return '|'.join([intent(question),tool,shape])

def score_map(folder,split):
 x=[x for x in rows(Path(folder)/'private_scores.jsonl') if x['split']==split];assert len(x)==1
 return {v['task_id']:bool(v['success']) for v in x[0]['case_results']},x[0]['metrics']

def first_calls(folder,scope):
 out={}
 for ep in rows(Path(folder)/'private_graph_traces.jsonl'):
  if ep['scope']!=scope:continue
  calls=[e for e in ep['events'] if e['node']=='execute'];assert ep['sample_id'] not in out
  out[ep['sample_id']]=calls[0] if calls else None
 return out

def fit(root):
 root=Path(root);p=json.loads((root/'protocol.json').read_text());data=json.loads((root/'private_learning.json').read_text());train={e['task_id']:e for e in data['train']}
 full,_=score_map(root/'fit_full','counterfactual_train');one,_=score_map(root/'fit_one','counterfactual_train');assert set(full)==set(one)==set(train) and len(train)==48
 a=first_calls(root/'fit_full','counterfactual_train:full');b=first_calls(root/'fit_one','counterfactual_train:one');assert set(a)==set(b)==set(train)
 groups=defaultdict(lambda:{'support':0,'one_gains':0,'one_losses':0,'full_correct':0,'one_correct':0});excluded={'missing_observation':0,'unmatched_first_observation':0,'ineligible_feature':0}
 for sid,e in train.items():
  if not a[sid] or not b[sid]:excluded['missing_observation']+=1;continue
  if any(a[sid][k]!=b[sid][k] for k in ['tool','args','result']):excluded['unmatched_first_observation']+=1;continue
  k=observation_key(e['question'],a[sid]['tool'],a[sid]['result'])
  if k is None:excluded['ineligible_feature']+=1;continue
  g=groups[k];g['support']+=1;g['one_gains']+=int(one[sid] and not full[sid]);g['one_losses']+=int(full[sid] and not one[sid]);g['full_correct']+=int(full[sid]);g['one_correct']+=int(one[sid])
 allowed=sorted(k for k,g in groups.items() if g['support']>=p['action_policy']['min_support'] and g['one_losses']==0)
 result={'allowed_keys':allowed,'training_groups':dict(groups),'excluded':excluded,'trained_on':48,'min_support':p['action_policy']['min_support'],'answer_identity_not_used':True,'selection_or_confirmation_used_for_fit':False}
 (root/'POLICY_CANDIDATE.json').write_text(json.dumps(result,indent=2));return result

def gate(root):
 root=Path(root);baseline,m0=score_map(root/'fit_full','selection');candidate,m1=score_map(root/'candidate','selection');assert set(baseline)==set(candidate) and len(baseline)==24
 h0,_=score_map(root/'fit_full','historical');h1,_=score_map(root/'candidate','historical');assert set(h0)==set(h1) and len(h0)==72
 gain=sum(candidate.values())-sum(baseline.values());loss=sum(h0[i] and not h1[i] for i in h0);reasons=[]
 if gain<1:reasons.append('current_correct_gain_below_one')
 if loss>2:reasons.append('historical_loss_budget_exceeded')
 if m1['format_compliance_rate']<.95:reasons.append('format_compliance_below_95pct')
 result={'accepted':not reasons,'reasons':reasons,'selection_before':sum(baseline.values()),'selection_after':sum(candidate.values()),'selection_total':24,'current_correct_gain':gain,'selection_format':m1['format_compliance_rate'],'historical_before':sum(h0.values()),'historical_after':sum(h1.values()),'historical_losses':loss,'historical_recoveries':sum(not h0[i] and h1[i] for i in h0),'historical_total':72}
 (root/'POLICY_GATE.json').write_text(json.dumps(result,indent=2));return result

def analyze(root):
 root=Path(root);p=json.loads((root/'protocol.json').read_text());result={'policy_candidate':json.loads((root/'POLICY_CANDIDATE.json').read_text()),'gate':json.loads((root/'POLICY_GATE.json').read_text()),'roles':{},'limitations':p['limitations'],'confirmation_opened':False}
 for role in ['fit_full','fit_one','candidate','confirm_full','confirm_policy']:
  d=root/role
  if not (d/'EXIT.json').exists():continue
  assert json.loads((d/'EXIT.json').read_text())['exit_code']==0
  metrics={x['split']:x['metrics'] for x in rows(d/'private_scores.jsonl')};traces=rows(d/'private_graph_traces.jsonl');calls=rows(d/'call_costs.jsonl');f=d/'private_feature_costs.jsonl';features=rows(f) if f.exists() else []
  result['roles'][role]={'metrics':metrics,'generation_calls':len(calls),'generation_seconds':sum(x['seconds'] for x in calls),'feature_forward_calls':len(features),'feature_seconds':sum(x['seconds'] for x in features),'tool_calls':sum(e['node']=='execute' for x in traces for e in x['events']),'learned_stops':sum(e['node']=='learned_stop' for x in traces for e in x['events'])}
 if result['gate']['accepted']:
  a,_=score_map(root/'confirm_policy','confirmation');b,_=score_map(root/'confirm_full','confirmation');ids={e['task_id'] for e in json.loads((root/'private_confirmation.json').read_text())};assert set(a)==set(b)==ids and len(ids)==100
  diff=[int(a[i])-int(b[i]) for i in sorted(ids)];g=diff.count(1);l=diff.count(-1);rng=random.Random(p['seed']);boot=sorted(sum(rng.choices(diff,k=100)) for _ in range(20000));n=g+l
  result['primary']={'gain':g,'loss':l,'delta_pp':sum(diff),'paired95_pp':[boot[499],boot[19499]],'mcnemar_p':min(1.,2*sum(math.comb(n,k) for k in range(min(g,l)+1))/2**n) if n else 1.};result['confirmation_opened']=True
 (root/'ANALYSIS.json').write_text(json.dumps(result,indent=2));return result
