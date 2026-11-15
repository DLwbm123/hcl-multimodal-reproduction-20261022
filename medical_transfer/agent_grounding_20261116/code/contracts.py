"""Compile training-only answer granularity and gate an independent-evidence harness."""
import json,re,random,math
from pathlib import Path
from collections import defaultdict
from adaptive import intent

def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]
def compile_contracts(train):
 assert len(train)==48 and all(e['split']=='train' and e['task_name']=='visual_qa' for e in train)
 vocab=defaultdict(set);support=defaultdict(int)
 for e in train:
  k=intent(e['question'])
  if k not in {'location','color'}:continue
  tokens=[x.strip().lower() for x in e['answer'].split(';')]
  if not tokens or any(not re.fullmatch(r'[a-z][a-z -]{0,39}',x) or x in {'nan','null'} for x in tokens):continue
  vocab[k].update(tokens);support[k]+=1
 return {'vocabulary':{k:sorted(v) for k,v in vocab.items()},'support':dict(support),'source':'48 parent training feedback cases only','selection_used':False,'confirmation_used':False}

def contract_for(question,artifact):
 k=intent(question);v=artifact.get('vocabulary',{}).get(k,[])
 if not v:return ''
 common='These are training-observed output labels, not evidence for the current image: '+json.dumps(v)+'. Use the CURRENT image to determine which apply; do not copy an example answer or infer presence from a label list. '
 if k=='location':return common+'Describe every visible region occupied by the requested target, not only its center or one region. The labels denote the corresponding top/middle/bottom and left/center/right image regions. If the target is absent, answer none. Separate multiple regions with semicolons.'
 return common+'Identify all visible colors of the requested target, not only the dominant color. Exclude unrelated surrounding tissue. If the target is absent, answer none. Separate multiple colors with semicolons.'

def score_map(folder,split):
 x=[x for x in rows(Path(folder)/'private_scores.jsonl') if x['split']==split];assert len(x)==1
 return {v['task_id']:bool(v['success']) for v in x[0]['case_results']},x[0]['metrics']

def gate(root):
 root=Path(root);a,m0=score_map(root/'eval_base','selection');b,m1=score_map(root/'eval_evidence','selection');assert set(a)==set(b) and len(a)==48
 h0,_=score_map(root/'eval_base','historical');h1,_=score_map(root/'eval_evidence','historical');assert set(h0)==set(h1) and len(h0)==72
 gain=sum(b.values())-sum(a.values());loss=sum(h0[i] and not h1[i] for i in h0);reasons=[]
 if gain<1:reasons.append('current_correct_gain_below_one')
 if loss>2:reasons.append('historical_loss_budget_exceeded')
 if m1['format_compliance_rate']<.95:reasons.append('format_compliance_below_95pct')
 x={'accepted':not reasons,'reasons':reasons,'selection_before':sum(a.values()),'selection_after':sum(b.values()),'selection_total':48,'current_correct_gain':gain,'selection_format':m1['format_compliance_rate'],'historical_before':sum(h0.values()),'historical_after':sum(h1.values()),'historical_losses':loss,'historical_recoveries':sum(not h0[i] and h1[i] for i in h0),'historical_total':72,'candidate_selection':'evidence only; contract ablation not eligible as fallback'}
 (root/'EVIDENCE_GATE.json').write_text(json.dumps(x,indent=2));return x

def analyze(root):
 root=Path(root);p=json.loads((root/'protocol.json').read_text());artifact=json.loads((root/'private_contracts.json').read_text());result={'gate':json.loads((root/'EVIDENCE_GATE.json').read_text()),'training_contract_support':artifact['support'],'training_vocabulary_counts':{k:len(v) for k,v in artifact['vocabulary'].items()},'roles':{},'limitations':p['limitations'],'confirmation_opened':False}
 for role in ['eval_base','eval_contract','eval_evidence','confirm_base','confirm_contract','confirm_evidence','confirm_shifted']:
  d=root/role
  if not (d/'EXIT.json').exists():continue
  assert json.loads((d/'EXIT.json').read_text())['exit_code']==0
  calls=rows(d/'call_costs.jsonl');features=rows(d/'private_feature_costs.jsonl');traces=rows(d/'private_graph_traces.jsonl')
  result['roles'][role]={'metrics':{x['split']:x['metrics'] for x in rows(d/'private_scores.jsonl')},'generation_calls':len(calls),'generation_seconds':sum(x['seconds'] for x in calls),'feature_forward_calls':len(features),'feature_seconds':sum(x['seconds'] for x in features),'tool_calls':sum(e['node']=='execute' for x in traces for e in x['events']),'completed_episodes':sum(x['status']=='completed' for x in traces),'failed_episodes':sum(x['status']!='completed' for x in traces)}
 if result['gate']['accepted']:
  expected={e['task_id'] for e in json.loads((root/'private_confirmation.json').read_text())};a,_=score_map(root/'confirm_evidence','confirmation');assert len(expected)==100 and set(a)==expected;result['paired_evidence_vs']={}
  for other in ['base','contract','shifted']:
   b,_=score_map(root/('confirm_'+other),'confirmation');assert set(b)==expected;diff=[int(a[i])-int(b[i]) for i in sorted(expected)];g=diff.count(1);l=diff.count(-1);rng=random.Random(p['seed']);boot=sorted(sum(rng.choices(diff,k=100)) for _ in range(20000));n=g+l
   result['paired_evidence_vs'][other]={'gain':g,'loss':l,'delta_pp':sum(diff),'paired95_pp':[boot[499],boot[19499]],'mcnemar_p':min(1.,2*sum(math.comb(n,k) for k in range(min(g,l)+1))/2**n) if n else 1.}
  result['confirmation_opened']=True
 (root/'ANALYSIS.json').write_text(json.dumps(result,indent=2));return result
