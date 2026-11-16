"""Frozen VLM/SAM primitives. Receives images only, never evaluator targets."""
import json,math,time
from pathlib import Path
import numpy as np
from PIL import Image
from core import bbox,decode,OPS

class FrozenModels:
 def __init__(self,p,out,guard):
  import torch
  from hcl.models import LocalTransformersModel
  from transformers import SamModel,SamProcessor
  self.p=p;self.out=Path(out);self.guard=guard;self.phase='preflight';self.scope='';self.seconds=0.
  offset_path=self.out.parent/'attempt_offsets.json'
  offsets=json.loads(offset_path.read_text()) if offset_path.exists() else {}
  self.calls=offsets.get('vlm',0);self.sam_calls=offsets.get('sam',0)
  torch.manual_seed(p['seed']);torch.set_num_threads(4)
  self.vlm=LocalTransformersModel(name='frozen',path=p['model_path'],torch_dtype='bfloat16',attn_implementation='sdpa',max_new_tokens=2048,answer_max_new_tokens=512,temperature=0.,enable_thinking=False,local_files_only=True)
  self.vlm._load()
  from integration import CoordinateBoundary
  self.boundary=CoordinateBoundary(self.vlm,self.out/'private_coordinate_trace.jsonl')
  self.sam=SamModel.from_pretrained(p['sam_path'],local_files_only=True).to('cuda').eval()
  self.processor=SamProcessor.from_pretrained(p['sam_path'],local_files_only=True)
  for model in [self.vlm._model,self.sam]:
   model.eval()
   for param in model.parameters():param.requires_grad_(False)
  self.assert_frozen()
 def assert_frozen(self):
  assert all(not m.training and all(not p.requires_grad for p in m.parameters()) for m in [self.vlm._model,self.sam])
 def emit(self,row):
  with (self.out/'private_calls.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 def generate(self,prompt,image=None,max_tokens=512,coordinate=False,role=None):
  self.guard()
  if self.calls>=self.p['budget']['max_total_model_calls']:raise RuntimeError('model call budget exceeded')
  self.calls+=1;t=time.monotonic();state={'router_phase':'skill_pilot','max_new_tokens':max_tokens}
  if image:state['task_context']={'files':[{'type':'image','path':str(image)}]}
  original_system=self.vlm.config.system_prompt
  if role=='controller':self.vlm.config.system_prompt='You select a registered tool or skill action. Valid tool names are ONLY localize, inspect, segment, measure, expand_box. Output exactly one JSON action object, for example {"tool":"localize","args":{}}. Never output a backend/model name as a tool name. Never directly answer the medical task in this role. Do not add an action wrapper or prose. A finish action is {"finish":true}; a separate node produces the answer.'
  elif role=='finalizer':self.vlm.config.system_prompt='You assemble a final structured answer from current tool evidence. Output only the requested JSON object, without markdown or tool calls. Copy coordinates and measurements from the ORIGINAL-image tool results; do not invent coordinates from the displayed resized image.'
  elif role=='proposal':self.vlm.config.system_prompt='You design reusable executable tool procedures from supplied training feedback. Output exactly one JSON skills object following the provided schema. Do not answer any example task.'
  try:
   if coordinate:
    with Image.open(image) as im:w,h=im.size
    key=f'view_{self.calls}'
    prepared=self.boundary.prepare_example({'task_name':'polyp_localization','task_id':key,'image':str(image),'visible_context':{'width':w,'height':h},'coordinate_goal':prompt})
    prompt=prepared['question'];state.update(router_phase='final_answer',internal_sample_id=key)
    text=self.boundary.generate(prompt,state=state)
   else:text=self.vlm.generate(prompt,state=state)
   with (self.out/'private_model_io.jsonl').open('a') as f:f.write(json.dumps({'call':self.calls,'phase':self.phase,'scope':self.scope,'image':str(image) if image else None,'prompt':prompt,'output':text})+'\n')
   self.emit({'kind':'vlm','call':self.calls,'phase':self.phase,'scope':self.scope,'seconds':time.monotonic()-t,'prompt_chars':len(prompt),'output_tokens':len(self.vlm._processor.tokenizer.encode(text)),'success':True})
   return text
  except BaseException:
   self.emit({'kind':'vlm','call':self.calls,'phase':self.phase,'scope':self.scope,'seconds':time.monotonic()-t,'success':False});raise
  finally:self.vlm.config.system_prompt=original_system
 def segment(self,image,box):
  import torch
  self.guard();t=time.monotonic();self.sam_calls+=1
  inputs=self.processor(images=image,input_boxes=[[box]],return_tensors='pt').to('cuda')
  with torch.inference_mode():outputs=self.sam(**inputs,multimask_output=True)
  masks=self.processor.image_processor.post_process_masks(outputs.pred_masks.cpu(),inputs['original_sizes'].cpu(),inputs['reshaped_input_sizes'].cpu())[0][0]
  scores=outputs.iou_scores[0,0].detach().float().cpu().numpy();idx=int(np.argmax(scores));mask=masks[idx].numpy().astype(bool)
  self.emit({'kind':'sam','call':self.sam_calls,'phase':self.phase,'scope':self.scope,'seconds':time.monotonic()-t,'selected_by':'predicted_iou','success':True})
  return mask,float(scores[idx])

class EpisodeTools:
 def __init__(self,models,image,out,limits):
  self.models=models;self.path=str(image);self.image=Image.open(image).convert('RGB');self.w,self.h=self.image.size;self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True);self.limits=limits;self.events=[];self.masks={};self.sams=0;self.initial_calls=models.calls
 def check(self):
  if len(self.events)>=self.limits['max_primitive_calls']:raise ValueError('primitive budget reached')
  if self.models.calls-self.initial_calls>=self.limits['max_vlm_calls_per_episode']-1:raise ValueError('episode VLM budget reached; one call reserved for finalizer')
 def execute(self,name,args,source='controller'):
  self.check();t=time.monotonic();event={'tool':name,'args':args,'source':source}
  try:
   if name not in OPS or not isinstance(args,dict) or set(args)-OPS[name]:raise ValueError('unknown tool or invalid arguments')
   result=self._execute(name,args);event['result']=result;return result
  except (ValueError,KeyError,TypeError,json.JSONDecodeError) as e:
   result={'status':'error','error':str(e)[:300]};event['result']=result;return result
  finally:
   event['seconds']=time.monotonic()-t;self.events.append(event)
 def _execute(self,name,a):
  if name in ['localize','inspect']:
   roi=bbox(a.get('roi',[0,0,self.w,self.h]) if name=='localize' else a['bbox'],self.w,self.h)
   x,y,x2,y2=roi;im=self.image.crop(tuple(roi));cw,ch=im.size
   if min(cw,ch)<28:raise ValueError('crop too small for visual backbone; minimum dimension 28')
   path=self.path
   if roi!=[0,0,self.w,self.h]:
    path=str(self.out/f'view_{len(self.events)}.png');im.save(path)
   if name=='inspect':
    question=a['question']
    if not isinstance(question,str) or not 1<=len(question)<=600:raise ValueError('invalid inspection question')
    text=self.models.generate('Describe only visual evidence in THIS CROP, including whether polyp tissue continues outside an image edge if visible. Do not use a historical answer. Question: '+question,path,256)
    return {'status':'ok','observation':text,'bbox':roi,'crop_size':[cw,ch]}
   instruction=a.get('instruction','Locate the visible polyp region(s).')
   if not isinstance(instruction,str) or len(instruction)>600:raise ValueError('invalid localization instruction')
   prompt='Locate the tight union box of the visible polyp tissue. '+instruction
   text=self.models.generate(prompt,path,self.limits['tool_max_tokens'],coordinate=True);d=decode(text);local=bbox(d['bbox'],cw,ch)
   mapped=[x+local[0],y+local[1],x+local[2],y+local[3]]
   mapped=bbox(mapped,self.w,self.h)
   return {'status':'ok','bbox':mapped,'view_bbox':roi,'crop_pixel_bbox':local,'coordinate_system':'original_pixel_half_open_xyxy'}
  if name=='expand_box':
   b=bbox(a['bbox'],self.w,self.h);f=a['fraction']
   if isinstance(f,bool) or not isinstance(f,(int,float)) or not 0<=f<=.5:raise ValueError('fraction must be 0..0.5')
   dx=(b[2]-b[0])*f;dy=(b[3]-b[1])*f
   return {'status':'ok','bbox':[max(0,math.floor(b[0]-dx)),max(0,math.floor(b[1]-dy)),min(self.w,math.ceil(b[2]+dx)),min(self.h,math.ceil(b[3]+dy))]}
  if name=='segment':
   box=bbox(a['bbox'],self.w,self.h)
   if self.sams>=self.limits['max_sam_calls_per_episode']:raise ValueError('SAM episode budget reached')
   self.sams+=1;mask,quality=self.models.segment(self.image,box)
   assert mask.shape==(self.h,self.w)
   mid=f'm{len(self.masks)+1}';self.masks[mid]=mask;Image.fromarray(mask.astype('uint8')*255).save(self.out/f'{mid}.png')
   ys,xs=np.where(mask);mb=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)] if len(xs) else None
   touches=bool(mb and (mb[0]<=box[0]+2 or mb[1]<=box[1]+2 or mb[2]>=box[2]-2 or mb[3]>=box[3]-2))
   return {'status':'ok','mask_id':mid,'bbox':mb,'prompt_bbox':box,'area_fraction':float(mask.mean()),'model_quality':quality,'touches_prompt_boundary':touches,'boundary_note':'Geometric signal only; not proof of truncation or accuracy.'}
  if name=='measure':
   mid=a['mask_id']
   if mid not in self.masks:raise ValueError('mask must be produced in CURRENT episode')
   m=self.masks[mid];ys,xs=np.where(m);mb=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)] if len(xs) else None
   return {'status':'ok','mask_id':mid,'bbox':mb,'foreground_pixels':int(m.sum()),'image_pixels':self.w*self.h,'area_fraction':float(m.mean())}
  raise ValueError('unknown primitive')
