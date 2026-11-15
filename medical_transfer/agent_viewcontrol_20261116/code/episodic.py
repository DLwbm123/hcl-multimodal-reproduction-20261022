"""Question-scoped, corrected training experiences; selection labels cannot enter retrieval."""
import json,re
from contextvars import ContextVar
from types import MethodType
from pathlib import Path

def question_key(q):
 return re.sub(r'\s+',' ',str(q).split('\n',1)[0].lower()).strip().rstrip('?. ')

def references(records,registry,question,current_id,limit=2):
 key=question_key(question);pool=[]
 for r in records:
  source=r.get('internal_source',{});sid=str(source.get('task_id',''));entry=registry.get(sid)
  if source.get('task_name')!='visual_qa' or not entry or sid==current_id:continue
  if question_key(r.get('question',''))!=key or question_key(entry['question'])!=key:continue
  answer=r.get('feedback')
  if not isinstance(answer,str) or not answer.strip():continue
  pool.append({'source_id':sid,'path':entry['image'],'question':entry['question'],'answer':answer.strip()})
 pool.sort(key=lambda x:x['source_id']);chosen=[];seen=set()
 for x in pool:
  if x['answer'].lower() not in seen:chosen.append(x);seen.add(x['answer'].lower())
  if len(chosen)==limit:return chosen
 for x in pool:
  if x not in chosen:chosen.append(x)
  if len(chosen)==limit:break
 return chosen

def install_scope(pipe,registry):
 active=ContextVar('visible_question',default=None);original_one=pipe._predict_one
 def one(self,stream_position,example,**kwargs):
  token=active.set((example.get('task_name'),question_key(example.get('question',''))))
  try:return original_one(stream_position,example,**kwargs)
  finally:active.reset(token)
 pipe._predict_one=MethodType(one,pipe)
 original=pipe.memory.raw.retrieval_candidates
 def candidates(self):
  rows=original();task,key=active.get() or (None,None)
  if task!='visual_qa':return rows
  ids={r['memory_id'] for r in self.records if str(r.get('internal_source',{}).get('task_id','')) in registry and question_key(r.get('question',''))==key}
  result=[]
  for r in rows:
   if r.get('memory_id') not in ids:continue
   x=dict(r);x.pop('previous_attempt',None);x.pop('previous_attempt_correct',None)
   x['usage_rule']='Corrected historical supervision for a different image. Infer the current answer from the current image.';result.append(x)
  return result
 pipe.memory.raw.retrieval_candidates=MethodType(candidates,pipe.memory.raw)

class ReferenceBackend:
 """Delegate every ordinary call; add bounded visual references only at finalization."""
 def __init__(self,backend,audit,max_side=448):
  self.backend=backend;self.config=backend.config;self.audit=Path(audit);self.max_side=max_side
 def __getattr__(self,name):return getattr(self.backend,name)
 def generate(self,prompt,*,state):
  refs=state.get('reference_examples',[])
  if state.get('router_phase')!='graph_finalize' or not refs:return self.backend.generate(prompt,state=state)
  assert len(refs)<=2
  from PIL import Image
  import torch
  b=self.backend;b._load();current=b._image_path_from_state(state);assert current
  content=[];images=[];visual=state.get('reference_images_enabled',True)
  for i,r in enumerate(refs,1):
   assert str(Path(r['path']).resolve())!=str(Path(current).resolve())
   content.append({'type':'text','text':f'Historical reference {i}, DIFFERENT IMAGE. Question: '+r['question']+'\nCorrected historical answer: '+r['answer']+'\nThis answer is not evidence for the current target.'})
   if visual:
    with Image.open(r['path']) as im:im=im.convert('RGB');im.thumbnail((self.max_side,self.max_side));images.append(im.copy())
    content.append({'type':'image','image':images[-1]})
  content.append({'type':'text','text':'CURRENT TARGET IMAGE follows. Answer only its question; compare visual evidence and do not transfer historical findings.'})
  with Image.open(current) as im:images.append(im.convert('RGB'))
  content.extend([{'type':'image','image':images[-1]},{'type':'text','text':prompt}])
  messages=[{'role':'system','content':self.config.system_prompt},{'role':'user','content':content}]
  proc=b._processor;model=b._model;text=proc.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=self.config.enable_thinking)
  inputs=proc(text=[text],images=images,return_tensors='pt');assert len(inputs['image_grid_thw'])==len(images)
  inputs=inputs.to(model.device)
  with torch.inference_mode():output=model.generate(**inputs,max_new_tokens=min(512,int(state.get('max_new_tokens',512))),do_sample=False)
  answer=proc.batch_decode(output[:,inputs.input_ids.shape[-1]:],skip_special_tokens=True)[0].strip()
  with self.audit.open('a') as f:f.write(json.dumps({'query_id':state.get('internal_sample_id'),'reference_ids':[r['source_id'] for r in refs],'visual':visual,'images':len(images),'target_last':True})+'\n')
  return answer
