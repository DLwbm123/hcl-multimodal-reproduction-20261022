"""Bounded data-only skill programs. No eval, code imports or external paths."""
import copy,json,math,re
OPS={'localize':{'roi','instruction'},'inspect':{'bbox','question'},'segment':{'bbox'},'measure':{'mask_id'},'expand_box':{'bbox','fraction'}}
CATALOG={
 'localize':'Locate the visible polyp union in the full image or an optional ROI. Arguments: roi optional [x1,y1,x2,y2], instruction optional string. Empty args {} is valid. Returns bbox in ORIGINAL pixels. Can be wrong.',
 'inspect':'Inspect a crop. Arguments: bbox [x1,y1,x2,y2], question string. Returns observation, original bbox, crop dimensions. Does NOT automatically revise bbox.',
 'segment':'Segment a region of the full original image. Arguments: bbox [x1,y1,x2,y2]. Returns mask_id, mask bbox, area_fraction, model_quality (not true accuracy), touches_prompt_boundary.',
 'measure':'args {mask_id:string from this episode}; returns area_fraction, foreground_pixels, image_pixels and mask bbox in original coordinates.',
 'expand_box':'args {bbox:[x1,y1,x2,y2],fraction:number 0..0.5}; symmetrically pads by fraction of box dimensions, clips to image; returns bbox.'}
TASKS={
 'T1':'Locate the visible polyp region(s). Return a tight union bounding box in ORIGINAL IMAGE pixels, half-open xyxy, as {"bbox":[x1,y1,x2,y2]}. Verify with tools when useful. The image contains a polyp; no category prediction is requested.',
 'T2':'Segment the visible polyp region(s) and measure its area as a fraction of the WHOLE ORIGINAL image. Return {"mask_id":"an actual mask produced in this episode","area_fraction":number}. Use actual image evidence, not historical coordinates or answers.'}
def decode(s):
 s=s.strip()
 if s.startswith('```'):
  s=s.split('\n',1)[1].rsplit('```',1)[0].strip()
 d=json.loads(s)
 if not isinstance(d,dict):raise ValueError('object required')
 return d

def decode_final(s,task):
 s=s.strip()
 if s.startswith('```'):s=s.split('\n',1)[1].rsplit('```',1)[0].strip()
 d=json.loads(s)
 if task=='T1':
  if isinstance(d,list) and len(d)==1:d=d[0]
  if isinstance(d,dict) and set(d)<= {'bbox','bbox_2d','label'} and ('bbox' in d)!=('bbox_2d' in d):return {'bbox':d.get('bbox',d.get('bbox_2d'))}
 if not isinstance(d,dict):raise ValueError('final answer must be an object')
 return d

def bbox(x,w,h):
 if not isinstance(x,list) or len(x)!=4 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in x):raise ValueError('bbox must contain four finite numbers')
 a,b,c,d=[int(round(v)) for v in x]
 if not(0<=a<c<=w and 0<=b<d<=h):raise ValueError('bbox outside original image or nonpositive')
 return [a,b,c,d]

def resolve(x,env):
 if isinstance(x,str) and x.startswith('$'):
  bits=x[1:].split('.');out=env
  for k in bits:
   if not isinstance(out,dict) or k not in out:raise ValueError('unavailable reference '+x)
   out=out[k]
  return copy.deepcopy(out)
 if isinstance(x,list):return [resolve(v,env) for v in x]
 if isinstance(x,dict):return {k:resolve(v,env) for k,v in x.items()}
 return x

def condition(c,env):
 if c is None:return True
 x=resolve(c['ref'],env);v=c['value'];op=c['op']
 if op=='eq':return x==v
 if not isinstance(x,(int,float)) or isinstance(x,bool):raise ValueError('numeric condition needs number')
 return x<float(v) if op=='lt' else x>float(v)

def validate_library(lib):
 if not isinstance(lib,list) or len(lib)>2:raise ValueError('skills must be a list of at most two')
 names=set()
 for sk in lib:
  serialized=json.dumps(sk)
  if re.search(r'\[\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*\]',serialized):raise ValueError('literal coordinates forbidden, including in prose')
  if re.search(r'(?:/data/|im_[a-f0-9]{12}|c_[a-f0-9]{12}|\.png|\.jpg)',serialized):raise ValueError('private paths/sample IDs forbidden in skills')
  required={'name','task','description','applies_when','steps','output'}
  if set(sk)!=required:raise ValueError('skill keys must be '+str(sorted(required)))
  if not re.fullmatch(r'[a-z][a-z0-9_]{2,47}',sk['name']) or sk['name'] in names:raise ValueError('bad/duplicate skill name')
  names.add(sk['name'])
  if sk['task'] not in TASKS:raise ValueError('unknown task')
  for k in ['description','applies_when']:
   if not isinstance(sk[k],str) or not 1<=len(sk[k])<=600:raise ValueError('invalid description/condition')
  if not isinstance(sk['steps'],list) or not 1<=len(sk['steps'])<=6:raise ValueError('1..6 steps required')
  known={'input','last'}
  def refs(x):
   if isinstance(x,str) and x.startswith('$'):
    if not re.fullmatch(r'\$[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+',x) or x[1:].split('.')[0] not in known:raise ValueError('invalid/forward reference '+x)
   elif isinstance(x,dict):
    for v in x.values():refs(v)
   elif isinstance(x,list):
    for v in x:refs(v)
  for step in sk['steps']:
   if set(step)-{'id','tool','args','when'} or not {'id','tool','args'}<=set(step):raise ValueError('step keys')
   sid=step['id'];op=step['tool'];a=step['args']
   if not isinstance(sid,str) or not re.fullmatch('[a-z][a-z0-9_]{0,23}',sid) or sid in known:raise ValueError('bad step id')
   if op not in OPS or not isinstance(a,dict) or set(a)-OPS[op]:raise ValueError('bad primitive/arguments')
   needed={'inspect':{'bbox','question'},'segment':{'bbox'},'measure':{'mask_id'},'expand_box':{'bbox','fraction'}}.get(op,set())
   if not needed<=set(a):raise ValueError('missing primitive arguments')
   for k in ['bbox','roi','mask_id']:
    if k in a and not(isinstance(a[k],str) and a[k].startswith('$')):raise ValueError('skill geometry/artifacts must use runtime references, not stored answers')
   if 'fraction' in a and (not isinstance(a['fraction'],(int,float)) or isinstance(a['fraction'],bool) or not 0<=a['fraction']<=.5):raise ValueError('fraction 0..0.5')
   for k in ['instruction','question']:
    if k in a and (not isinstance(a[k],str) or len(a[k])>600):raise ValueError('tool text limit')
   refs(a)
   if 'when' in step:
    c=step['when']
    if not isinstance(c,dict) or set(c)!={'ref','op','value'} or c['op'] not in ['eq','lt','gt'] or not isinstance(c['ref'],str) or not c['ref'].startswith('$'):raise ValueError('invalid condition')
    if isinstance(c['value'],(list,dict)) or c['value'] is None:raise ValueError('condition requires scalar')
    refs(c)
   known.add(sid)
  expected={'bbox'} if sk['task']=='T1' else {'mask_id','area_fraction'}
  if not isinstance(sk['output'],dict) or set(sk['output'])!=expected:raise ValueError('output contract')
  if any(not isinstance(v,str) or not v.startswith('$') for v in sk['output'].values()):raise ValueError('outputs must be runtime references')
  refs(sk['output'])
  if len(json.dumps(sk))>7000:raise ValueError('skill exceeds bytes')
 return copy.deepcopy(lib)

SCHEMA='''Return {"skills":[...]} with at most two skills. No code or markdown.
Every skill has exactly name (lower_snake_case), task (T1 or T2), description, applies_when, steps, output.
steps is 1..6 objects {"id":"a","tool":"primitive_name","args":{...}}.
Optional when={"ref":"$a.field","op":"eq|lt|gt","value":scalar} skips a step when false. Skipped step id aliases last successful result. No loops, recursion, tools outside catalog, or file paths.
Runtime references are strings "$input.full_bbox", "$a.bbox", "$a.mask_id", "$a.area_fraction", "$last.bbox" etc. References may only address input, last, or previous steps. input includes full_bbox,width,height. last initially includes bbox=full_bbox.
Literal bounding boxes, historical mask IDs and literal final answers are forbidden; derive geometry from current observations.
T1 output exactly {"bbox":"$some_step.bbox"}; T2 output exactly {"mask_id":"$some_step.mask_id","area_fraction":"$some_step.area_fraction"}.
Each step consumes the same primitive budget as an ordinary tool call. A skill call returns its proposed output to the controller; it does not automatically finalize.
Generate procedures justified by observed training failures, not generic reminders. You may add, retain or revise skills; an empty library is allowed. Existing skills can be lost if omitted, so preserve useful ones. No prewritten candidate workflow is supplied.'''
