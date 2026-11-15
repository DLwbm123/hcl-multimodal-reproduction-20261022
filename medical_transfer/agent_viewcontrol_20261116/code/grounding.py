"""A bounded region-proposal/crop skill, selected only by training evidence."""
import json,math
from pathlib import Path
from vision_tools import VisionTools
from adaptive import intent

ELIGIBLE={'color','location'}
def use_macro(question,arm,policy):
 k=intent(question)
 return k in ELIGIBLE and (arm=='macro' or (arm=='proposed' and k in policy.get('allowed_macro_intents',[])))

def parse_region(raw,size):
 text=raw.strip()
 if text.startswith('```json') and text.endswith('```'):text=text[7:-3].strip()
 elif text.startswith('```') and text.endswith('```'):text=text[3:-3].strip()
 value=json.loads(text);assert isinstance(value,dict) and set(value)=={'present','bbox'},'region requires present/bbox only'
 present=value['present'];assert present is None or type(present) is bool,'present must be boolean or null'
 if present is not True:
  assert value['bbox'] is None,'absent/uncertain proposal cannot contain box'
  return {'status':'absent' if present is False else 'uncertain','crop_available':False}
 box=value['bbox'];assert isinstance(box,list) and len(box)==4 and all(type(v) is int for v in box),'four integer normalized coordinates required'
 assert 0<=box[0]<box[2]<=1000 and 0<=box[1]<box[3]<=1000,'invalid normalized geometry'
 w,h=size;assert min(w,h)>=28,'image below crop processor size'
 x1,y1,x2,y2=math.floor(box[0]*w/1000),math.floor(box[1]*h/1000),math.ceil(box[2]*w/1000),math.ceil(box[3]*h/1000)
 cw=min(w,max(28,math.ceil((x2-x1)*1.2)));ch=min(h,max(28,math.ceil((y2-y1)*1.2)))
 left=max(0,min(w-cw,math.floor((x1+x2-cw)/2)));top=max(0,min(h-ch,math.floor((y1+y2-ch)/2)));crop=[left,top,left+cw,top+ch]
 assert 0<=crop[0]<crop[2]<=w and 0<=crop[1]<crop[3]<=h
 return {'status':'ok','crop_available':True,'bbox_normalized_0_1000':box,'target_bbox_original':[x1,y1,x2,y2],'crop_bbox_original':crop,'original_size':[w,h],'proposal_source':'frozen VLM, unvalidated','interpretation':'A proposal, not target ground truth. Crop is padded by 10 percent of box extent per side, with minimum 28 pixels.'}

class GroundingTools(VisionTools):
 # The primitive is internal to the skill, not added to the incumbent tool menu.
 def validate(self,name,args):
  if name=='propose_target_region':
   if args!={}:raise ValueError('region proposal has no model-supplied arguments')
   return
  return super().validate(name,args)
 def propose(self,state,backend):
  image=self.load_image(state);question=state['question'].split('\n',1)[0]
  prompt=('Identify the visual target requested by the CURRENT QUESTION in the CURRENT image. Propose the smallest axis-aligned box enclosing all visible instances of that target, not the whole background. Do not answer its color or location question yet. '
   'Output exactly one JSON object with keys present and bbox. For a visible target use {"present":true,"bbox":[x1,y1,x2,y2]}, integer coordinates normalized to 0..1000, origin upper-left, x horizontal and y vertical. '
   'If absent use {"present":false,"bbox":null}. If uncertain use {"present":null,"bbox":null}. Do not assume the question implies presence. No prose or other keys.\nCURRENT QUESTION:\n'+question)
  raw=backend.generate(prompt,state={**state,'question':question,'router_phase':'graph_tool','max_new_tokens':256})
  try:result=parse_region(raw,image.size)
  except (AssertionError,ValueError,TypeError,KeyError) as exc:result={'status':'error','crop_available':False,'error':str(exc)[:180]}
  with (self.output_dir/'private_geometry.jsonl').open('a') as f:f.write(json.dumps({'raw_detector':raw,'result':result})+'\n')
  return result

import random
from collections import defaultdict
from contracts import rows,score_map

def fit(root):
 root=Path(root);p=json.loads((root/'protocol.json').read_text());train=json.loads((root/'private_learning.json').read_text())['train'];assert len(train)==48 and all(e['split']=='train' for e in train)
 a,_=score_map(root/'fit_incumbent','counterfactual_train');b,_=score_map(root/'fit_macro','counterfactual_train');assert set(a)==set(b)=={e['task_id'] for e in train}
 groups=defaultdict(lambda:{'support':0,'incumbent_correct':0,'macro_correct':0,'gains':0,'losses':0})
 for e in train:
  k=intent(e['question']);sid=e['task_id'];v=groups[k];v['support']+=1;v['incumbent_correct']+=a[sid];v['macro_correct']+=b[sid];v['gains']+=int(b[sid] and not a[sid]);v['losses']+=int(a[sid] and not b[sid])
 rule=p['routing'];allowed=sorted(k for k,v in groups.items() if k in ELIGIBLE and v['support']>=rule['min_support'] and v['gains']-v['losses']>=rule['min_net_training_gain'] and v['losses']<=rule['max_training_losses'])
 result={'allowed_macro_intents':allowed,'training_groups':dict(groups),'training_count':48,'selection_used':False,'confirmation_used':False,'fallback':'unchanged incumbent','incumbent':json.loads((root/'INCUMBENT_POLICY.json').read_text()),'rule':{k:rule[k] for k in ['min_support','min_net_training_gain','max_training_losses']}};(root/'SKILL_CANDIDATE.json').write_text(json.dumps(result,indent=2));return result

def gate(root):
 root=Path(root);a,_=score_map(root/'eval_incumbent','selection');b,m=score_map(root/'eval_proposed','selection');h0,_=score_map(root/'eval_incumbent','historical');h1,_=score_map(root/'eval_proposed','historical');assert len(a)==24 and set(a)==set(b) and len(h0)==72 and set(h0)==set(h1)
 gain=sum(b.values())-sum(a.values());loss=sum(h0[k] and not h1[k] for k in h0);reasons=[]
 if gain<1:reasons.append('current_correct_gain_below_one')
 if loss>2:reasons.append('historical_loss_budget_exceeded')
 if m['format_compliance_rate']<.95:reasons.append('format_compliance_below_95pct')
 result={'accepted':not reasons,'reasons':reasons,'selection_before':sum(a.values()),'selection_after':sum(b.values()),'selection_total':24,'current_correct_gain':gain,'paired_gains':sum(b[k] and not a[k] for k in a),'paired_losses':sum(a[k] and not b[k] for k in a),'selection_format':m['format_compliance_rate'],'historical_before':sum(h0.values()),'historical_after':sum(h1.values()),'historical_losses':loss,'historical_recoveries':sum(h1[k] and not h0[k] for k in h0),'historical_total':72};(root/'SKILL_GATE.json').write_text(json.dumps(result,indent=2));return result

def analyze(root):
 root=Path(root);policy=json.loads((root/'SKILL_CANDIDATE.json').read_text());g=json.loads((root/'SKILL_GATE.json').read_text());out={'policy':policy,'gate':g,'confirmation_opened':False,'roles':{}}
 roles=['fit_incumbent','fit_macro']+(['eval_incumbent','eval_macro','eval_proposed'] if policy['allowed_macro_intents'] else [])+(['confirm_incumbent','confirm_proposed','confirm_macro'] if g['accepted'] else [])
 for role in roles:
  d=root/role;assert json.loads((d/'EXIT.json').read_text())['exit_code']==0;calls=rows(d/'call_costs.jsonl');assert len(calls)==json.loads((d/'model_attempts.json').read_text())['attempts'];features=rows(d/'private_feature_costs.jsonl');traces=rows(d/'private_graph_traces.jsonl');audit=rows(d/'private_retrieval_audit.jsonl');scores=rows(d/'private_scores.jsonl');ro=rows(d/'read_only_checks.jsonl');assert len(ro)==len(scores) and all(x['unchanged'] and x['weights_frozen'] for x in ro)
  out['roles'][role]={'metrics':{x['split']:x['metrics'] for x in scores},'generation_calls':len(calls),'generation_failed':sum(not x['success'] for x in calls),'generation_seconds':sum(x['seconds'] for x in calls),'feature_forwards':len(features),'feature_seconds':sum(x['seconds'] for x in features),'graph_completed':sum(x['status']=='completed' for x in traces),'graph_failed':sum(x['status']!='completed' for x in traces),'tool_calls':sum(ev['node']=='execute' for x in traces for ev in x['events']),'macro_queries':sum(x['grounded_macro'] for x in audit),'geometry_proposals':{k:sum(x['result']['status']==k for x in rows(d/'private_tool_artifacts/private_geometry.jsonl')) if (d/'private_tool_artifacts/private_geometry.jsonl').exists() else 0 for k in ['ok','absent','uncertain','error']},'read_only_and_weights_frozen':True}
 if g['accepted']:
  p=json.loads((root/'protocol.json').read_text());a,_=score_map(root/'confirm_proposed','confirmation');ids={x['task_id'] for x in json.loads((root/'private_confirmation.json').read_text())};assert len(ids)==76 and set(a)==ids;out['paired_proposed_vs']={}
  for role in ['incumbent','macro']:
   b,_=score_map(root/('confirm_'+role),'confirmation');assert set(b)==ids;diff=[int(a[k])-int(b[k]) for k in sorted(ids)];gain=diff.count(1);loss=diff.count(-1);n=gain+loss;rng=random.Random(p['seed']);boot=sorted(100*sum(rng.choices(diff,k=76))/76 for _ in range(20000));out['paired_proposed_vs'][role]={'gain':gain,'loss':loss,'delta_pp':100*sum(diff)/76,'paired95_pp':[boot[499],boot[19499]],'mcnemar_p':min(1.,2*sum(math.comb(n,k) for k in range(min(gain,loss)+1))/2**n) if n else 1.}
  out['confirmation_opened']=True
 (root/'ANALYSIS.json').write_text(json.dumps(out,indent=2));return out
