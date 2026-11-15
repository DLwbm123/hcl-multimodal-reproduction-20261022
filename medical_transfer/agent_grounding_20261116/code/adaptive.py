"""Frozen visible-feature applicability and train-only, rollback-bound tool evidence."""
import json,re,time
from pathlib import Path
from contextvars import ContextVar
from types import MethodType
from collections import defaultdict
import numpy as np
from episodic import question_key

VALID_TOOLS={'classify_image','localize_region','answer_visual_question','inspect_region','analyze_image_quality','segmentation'}
def intent(q):
 q=question_key(q)
 if 'how many' in q or 'number of' in q:return 'count'
 if 'color' in q or 'colour' in q:return 'color'
 if q.startswith('where') or 'location' in q:return 'location'
 if q.startswith(('is ','are ','does ','do ','can ')):return 'presence'
 if 'instrument' in q and ('what' in q or 'which' in q):return 'instrument_type'
 return 'other'
def objects(q):
 q=question_key(q);found=set()
 for group,terms in {'instrument':['instrument','instrumnet','surgical tool'],'polyp':['polyp'],'abnormality':['abnormal'],'finding':['finding'],'landmark':['landmark'],'artifact':['artifact','artefact']}.items():
  if any(t in q for t in terms):found.add(group)
 return found

def eligible(question,source_question,text_similarity,threshold):
 if question_key(question)==question_key(source_question):return True
 a,b=intent(question),intent(source_question)
 if a=='other' or a!=b:return False
 x,y=objects(question),objects(source_question)
 if x!=y:return False
 return text_similarity>=threshold

class AdaptiveIndex:
 def __init__(self,backend,registry,audit,guard):
  self.backend=backend;self.registry=registry;self.audit=Path(audit);self.guard=guard;self.text_cache={};self.image_cache={};self.threshold=None
 def record(self,kind,start,key):
  with self.audit.open('a') as f:f.write(json.dumps({'kind':kind,'seconds':time.monotonic()-start,'key':key,'weights_frozen':True})+'\n')
 def text(self,q):
  key=question_key(q)
  if key not in self.text_cache:
   self.guard();start=time.monotonic();import torch
   b=self.backend;b._load();tokens=b._processor.tokenizer(key,return_tensors='pt',truncation=True,max_length=128).to(b._model.device)
   with torch.inference_mode():
    h=b._model.model(input_ids=tokens.input_ids,attention_mask=tokens.attention_mask,use_cache=False,return_dict=True).last_hidden_state[0]
    m=tokens.attention_mask[0].bool();v=h[m].float().mean(0);v=v/v.norm().clamp_min(1e-12)
   self.text_cache[key]=v.cpu().numpy();self.record('text_forward',start,key)
  return self.text_cache[key]
 def image(self,path):
  key=str(Path(path).resolve())
  if key not in self.image_cache:
   self.guard();start=time.monotonic();import torch
   from PIL import Image
   b=self.backend;b._load()
   with Image.open(path) as im:im=im.convert('RGB');im.thumbnail((448,448));pixels=b._processor.image_processor(images=[im],return_tensors='pt')
   with torch.inference_mode():
    h=b._model.visual(pixels['pixel_values'].to(device=b._model.device,dtype=b._model.visual.dtype),grid_thw=pixels['image_grid_thw'].to(b._model.device));v=h.float().mean(0);v=v/v.norm().clamp_min(1e-12)
   self.image_cache[key]=v.cpu().numpy();self.record('vision_forward',start,key)
  return self.image_cache[key]
 def ensure(self):
  if self.threshold is not None:return
  entries=list(self.registry.values())[:24];vectors=[self.text(e['question']) for e in entries]
  negative=[float(vectors[i]@vectors[j]) for i in range(len(entries)) for j in range(i) if intent(entries[i]['question'])!=intent(entries[j]['question'])]
  self.threshold=float(np.quantile(negative,.95)) if negative else 1.0
  for e in entries:self.image(e['image'])
  self.audit.with_name('FEATURE_CALIBRATION.json').write_text(json.dumps({'threshold':self.threshold,'negative_pairs':len(negative),'source':'visible_training_only','training_images':len(entries),'no_labels_or_outcome_selection':True},indent=2))
 def ranked(self,records,question,image,current_id):
  pool=[r for r in records if r.get('internal_source',{}).get('task_name')=='visual_qa' and str(r.get('internal_source',{}).get('task_id')) in self.registry and str(r['internal_source']['task_id'])!=current_id]
  if not pool:return []
  self.ensure();qt=self.text(question);qi=self.image(image);out=[]
  for r in pool:
   sid=str(r['internal_source']['task_id']);e=self.registry[sid];sim=float(qt@self.text(e['question']))
   if not eligible(question,e['question'],sim,self.threshold):continue
   visual=float(qi@self.image(e['image']));out.append((.7*sim+.3*visual,sid,r))
  return sorted(out,key=lambda x:(-x[0],x[1]))
 def references(self,records,question,image,current_id):
  ranked=self.ranked(records,question,image,current_id);pool=[]
  for score,sid,r in ranked:
   answer=r.get('feedback');e=self.registry[sid]
   if not isinstance(answer,str) or not answer.strip():continue
   pool.append({'source_id':sid,'path':e['image'],'question':e['question'],'answer':answer.strip()})
  chosen=[];seen=set()
  for x in pool:
   if x['answer'].lower() not in seen:chosen.append(x);seen.add(x['answer'].lower())
   if len(chosen)==2:return chosen
  for x in pool:
   if x not in chosen:chosen.append(x)
   if len(chosen)==2:break
  return chosen

def install_adaptive_scope(pipe,index):
 active=ContextVar('query_visible',default=None);original_one=pipe._predict_one
 def one(self,pos,e,**kwargs):
  token=active.set(e)
  try:return original_one(pos,e,**kwargs)
  finally:active.reset(token)
 pipe._predict_one=MethodType(one,pipe);original=pipe.memory.raw.retrieval_candidates
 def candidates(self):
  rows=original();e=active.get()
  if not e or e.get('task_name')!='visual_qa':return rows
  ranked=index.ranked(self.records,e['question'],e['image'],str(e['task_id']));allowed={r['memory_id'] for _,_,r in ranked}
  result=[]
  for row in rows:
   if row.get('memory_id') not in allowed:continue
   v=dict(row);v.pop('previous_attempt',None);v.pop('previous_attempt_correct',None);v['usage_rule']='Corrected historical supervision for a different image; current visual evidence takes precedence.';result.append(v)
  return result
 pipe.memory.raw.retrieval_candidates=MethodType(candidates,pipe.memory.raw)

def attach_training_evidence(pipe,examples,episodes,result):
 from medical_hcl.evaluation import normalize
 assert all(e.get('split')=='train' and e.get('task_name')=='visual_qa' for e in examples),'tool evidence accepts training VQA only'
 examples={str(e['task_id']):e for e in examples};scores={str(x['task_id']):bool(x['success']) for x in result['case_results']};evidence={}
 for ep in episodes:
  sid=str(ep.get('sample_id'));assert sid in examples and sid in scores
  calls=[x for x in ep['events'] if x['node']=='execute'];names=[x['tool'] for x in calls];assert set(names)<=VALID_TOOLS
  first_correct=bool(calls) and normalize(calls[0]['result'])==normalize(examples[sid]['answer'])
  evidence[sid]={'intent':intent(examples[sid]['question']),'tools':names,'first_correct':first_correct,'final_correct':scores[sid]}
 for r in pipe.memory.raw.records:
  sid=str(r.get('internal_source',{}).get('task_id'))
  if sid in evidence:r['execution_experience']=evidence[sid]
 pipe.memory.raw._persist()

def tool_evidence(records,registry,question,current_id):
 if intent(question)=='other':return {}
 groups=defaultdict(lambda:{'episodes':0,'final_correct':0,'first_observation_correct':0})
 for r in records:
  sid=str(r.get('internal_source',{}).get('task_id'));e=r.get('execution_experience')
  if sid not in registry or sid==current_id or not e or e['intent']!=intent(question):continue
  names=e['tools']
  if not names or len(names)>2 or not set(names)<=VALID_TOOLS:continue
  g=groups[tuple(names)];g['episodes']+=1;g['final_correct']+=int(e['final_correct']);g['first_observation_correct']+=int(e['first_correct'])
 patterns=[{'tools':list(k),**v} for k,v in groups.items() if v['episodes']>=2]
 patterns.sort(key=lambda p:(-p['episodes'],p['tools']))
 return {'question_intent':intent(question),'patterns':patterns[:3]} if patterns else {}
