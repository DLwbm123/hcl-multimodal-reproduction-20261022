import os,sys,json,time,copy,traceback,math
from pathlib import Path
R=Path(os.environ['RUN_ROOT']);P=json.loads((R/'protocol.json').read_text());B=json.loads((R/'budget.json').read_text());O=R/'run';O.mkdir(exist_ok=True)
sys.path[:0]=[str(R/'code')]+P['python_paths']
os.environ.update(HF_HOME=str(R/'cache'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TMPDIR=str(R/'tmp'),TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',CUDA_CACHE_PATH=str(R/'cache/cuda'),XDG_CACHE_HOME=str(R/'cache'),TORCH_HOME=str(R/'cache/torch'),TRITON_CACHE_DIR=str(R/'cache/triton'),PYTHONDONTWRITEBYTECODE='1')
import tempfile
tempfile.tempdir=str(R/'tmp')
from core import TASKS,CATALOG,SCHEMA,decode,validate_library
from runtime import FrozenModels,EpisodeTools
from agent import GraphAgent,fixed_run
from evaluate import score,aggregate,accept

def write(name,x):
 p=O/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)
def append(name,x):
 with (O/name).open('a') as f:f.write(json.dumps(x,allow_nan=False)+'\n')
def guard():
 if (R/'HARD_STOP.json').exists() or time.time()>=B['soft_deadline_epoch']:raise RuntimeError('original deadline or external stop')
 f=R/'resource_status.json';s=json.loads(f.read_text())
 if time.time()-s['time']>120:raise RuntimeError('stale independent resource guard')

def memory_for(task,raw):
 # Fixed first two in source order, not selected by outcome. Both full and no-skill arms get identical records.
 selected=copy.deepcopy(raw.get(task,[])[:P['learning']['raw_examples_per_query']])
 while len(json.dumps(selected))>P['learning']['raw_chars']:
  longest=max(selected,key=lambda x:len(json.dumps(x)))
  if longest.get('tool_trajectory'):
   longest['tool_trajectory'].pop();longest['trajectory_truncated']=True
  else:selected.pop()
 return selected

def run_set(task,cases,state,label,fixed=False):
 before=copy.deepcopy(state);result=[]
 for i,case in enumerate(cases):
  guard();models.phase=label;models.scope=f'{task}:{i}';out=O/'private_artifacts'/label/task/f'{i:03d}'
  tools=EpisodeTools(models,case['image'],out,P['runtime'])
  episode=fixed_run(task,tools) if fixed else agent.run(task,tools,state['skills'],memory_for(task,state['raw']))
  metrics=score(task,case,episode,tools,P['evaluation'])
  row={'id':case['id'],'group_id':case['group_id'],'task':task,'phase':label,'episode':episode,'metrics':metrics,'artifact_dir':str(out)}
  result.append(row);append('private_episodes.jsonl',row)
  write('progress.json',{'phase':label,'task':task,'completed':i+1,'total':len(cases),'vlm_calls':models.calls,'sam_calls':models.sam_calls,'skill_count':len(state['skills'])})
 assert before==state,'evaluation altered persistent state'
 models.assert_frozen();append('readonly_receipts.jsonl',{'phase':label,'task':task,'n':len(cases),'state_unchanged':True,'weights_frozen':True})
 return result

def feedback(row,case):
 m=row['metrics'];ep=row['episode']
 f={'success':m['success'],'score':m['score'],'invalid_reason':m.get('invalid_reason'),'target_bbox':case['bbox']}
 if row['task']=='T2':
  from PIL import Image
  import numpy as np
  f.update(target_area_fraction=float((np.asarray(Image.open(case['mask']).convert('L'))>127).mean()),dice=m['dice'],area_error=m['area_error'])
 return {'task':row['task'],'image_size':[case['width'],case['height']],'tool_trajectory':[{'tool':e['tool'],'args':e['args'],'result':e['result']} for e in ep['tools']],'answer':ep['answer'],'feedback':f}

def propose(task,state,evidence,index,prior_errors,previous_proposals):
 instruction=('Learn reusable executable skills for medical image analysis from TRAINING feedback only. '
  'The evaluator, input images and models cannot be changed. Fix concrete procedural mistakes, do not memorize geometry or labels. '
  'You may inspect a crop, change localization instructions, measure a real mask, or revise a tool sequence/condition when supported by these trajectories. '
  'Do not invent clinical facts or interpret a predicted mask-quality score as ground-truth Dice. '
  'Use current-image runtime variables. A shorter correct process is preferable to redundant calls. '
  'No particular sequence is mandatory. '+SCHEMA+'\nTOOLS:\n'+json.dumps(CATALOG)+'\nCURRENT TASK:\n'+TASKS[task]+
  '\nEXISTING LIBRARY:\n'+json.dumps(state['skills'])+'\nTRAINING EPISODES AND FEEDBACK:\n'+json.dumps(evidence)+'\nPrior candidate schema errors (not validation/test answers):\n'+json.dumps(prior_errors)+
  '\nEarlier proposed libraries, without validation outcomes:\n'+json.dumps(previous_proposals)+
  '\nProposal number '+str(index+1)+'. If earlier proposals exist, offer a materially different training-evidence-based alternative rather than repeating them.')
 last_error=[]
 for repair in range(P['learning']['proposal_format_repairs']+1):
  models.phase='skill_proposal';models.scope=f'{task}:{index}:{repair}'
  output=models.generate(instruction,max_tokens=P['runtime']['proposal_max_tokens'],role='proposal')
  record={'task':task,'candidate':index,'repair':repair,'prompt':instruction,'output':output}
  try:
   d=decode(output)
   if set(d)!={'skills'}:raise ValueError('root key must be skills')
   lib=validate_library(d['skills']);record['valid']=True;append('private_proposals.jsonl',record);return lib,last_error
  except (ValueError,KeyError,TypeError) as e:
   last_error.append(str(e));record.update(valid=False,error=str(e));append('private_proposals.jsonl',record)
   instruction+='\nYour JSON failed validation: '+str(e)+'. Return a corrected full library. Previous output:\n'+output
 return None,last_error

try:
 write('START.json',{'pid':os.getpid(),'started':time.time(),'original_start':B['started_epoch']})
 models=FrozenModels(P,O,guard);agent=GraphAgent(models,P['runtime']);write('MODEL_READY.json',{'frozen':True,'vlm':P['model_path'],'segmentation':P['sam_path']})
 data=json.loads((R/'private_data.json').read_text())
 # Engineering + capability check ONLY on excluded preflight cases. Gold boxes are never runtime tool inputs later.
 preflight=[]
 for i,case in enumerate(data['preflight']):
  tools=EpisodeTools(models,case['image'],O/'private_artifacts'/'preflight'/str(i),P['runtime']);loc=tools.execute('localize',{})
  seg=tools.execute('segment',{'bbox':case['bbox']},source='oracle_diagnostic_only')
  if seg.get('status')!='ok':raise RuntimeError('SAM preflight failed '+str(seg))
  measure=tools.execute('measure',{'mask_id':seg['mask_id']})
  ep={'answer':{'mask_id':seg['mask_id'],'area_fraction':measure['area_fraction']},'error':''}
  met=score('T2',case,ep,tools,P['evaluation']);preflight.append({'id':case['id'],'localize':loc,'oracle_sam_metrics':met,'tools':tools.events})
  write('private_preflight.json',preflight)
  if loc.get('status')!='ok':raise RuntimeError('real localization interface preflight failed')
 write('private_preflight.json',preflight)
 capability=any(x['oracle_sam_metrics']['dice']>=.70 for x in preflight)
 write('PREFLIGHT.json',{'passed':capability,'oracle_box_only_diagnostic':True,'oracle_dice':[x['oracle_sam_metrics']['dice'] for x in preflight]})
 if not capability:raise RuntimeError('CAPABILITY_BLOCK: neither preflight oracle-box SAM mask reached Dice .70; no skill study started')
 graph_checks=[]
 for task,case in zip(P['tasks'],data['preflight']):
  tools=EpisodeTools(models,case['image'],O/'private_artifacts'/'graph_preflight'/task,P['runtime'])
  episode=agent.run(task,tools,[],[]);metrics=score(task,case,episode,tools,P['evaluation'])
  graph_checks.append({'task':task,'episode':episode,'metrics':metrics})
 write('private_graph_preflight.json',graph_checks)
 graph_ok=all(x['metrics']['valid'] and len(x['episode']['tools'])>0 for x in graph_checks)
 write('GRAPH_PREFLIGHT.json',{'passed':graph_ok,'valid_outputs':[x['metrics']['valid'] for x in graph_checks],'accuracy_not_used':True})
 if not graph_ok:raise RuntimeError('INTERFACE_BLOCK: graph did not produce valid grounded outputs on both excluded preflight tasks')
 state={'skills':[],'raw':{}};states={'S0':copy.deepcopy(state)};gates=[];train_results={};val_results={}
 for task in P['learning']['stages']:
  # Train and validation/test roles are separate; no feedback enters inference during validation.
  train_rows=run_set(task,data['train'],state,f'train_{task}')
  evidence=[feedback(r,c) for r,c in zip(train_rows,data['train'])];train_results[task]=train_rows
  # Keep task-relevant raw trajectory examples identical between skill candidate and its no-skill counterfactual.
  raw=[]
  for e in evidence:
   e=copy.deepcopy(e)
   for step in e['tool_trajectory']:
    if len(json.dumps(step))>1200:step['result']={'status':step['result'].get('status'),'summary':'long observation omitted'}
   raw.append(e)
  state['raw'][task]=raw
  current=run_set(task,data['validation'],state,f'gate_{task}_before')
  old=run_set('T1',data['validation'],state,f'gate_{task}_history_before') if task=='T2' else []
  errors=[];previous_proposals=[]
  for idx in range(P['learning']['candidates_per_stage']):
   lib,errors=propose(task,state,evidence,idx,errors,previous_proposals)
   if lib is None:
    rec={'task':task,'candidate':idx,'accepted':False,'reasons':['invalid_skill_schema'],'errors':errors};gates.append(rec);append('gate_decisions.jsonl',rec);continue
   if lib in previous_proposals:
    rec={'task':task,'candidate':idx,'accepted':False,'reasons':['duplicate_candidate']};gates.append(rec);append('gate_decisions.jsonl',rec);continue
   previous_proposals.append(copy.deepcopy(lib))
   candidate=copy.deepcopy(state);candidate['skills']=lib
   proposed=run_set(task,data['validation'],candidate,f'gate_{task}_candidate{idx}')
   old_after=run_set('T1',data['validation'],candidate,f'gate_{task}_history{idx}') if task=='T2' else []
   rec={'task':task,'candidate':idx,**accept(current,proposed,old,old_after,P['gate'])}
   rec['proposed_skill_names']=[s['name'] for s in lib];gates.append(rec);append('gate_decisions.jsonl',rec)
   write(f'candidate_{task}_{idx}.json',{'library':lib,'gate':rec})
   if rec['accepted']:
    # Historical budget is cumulative relative to the committed end-of-T1 state.
    state=candidate;current=proposed
   # No validation per-case labels or errors are supplied to the proposer.
  val_results[task]=aggregate(current)
  states['S1' if task=='T1' else 'S2']=copy.deepcopy(state)
  write(f'state_{task}.json',state)
 states['RAW']={'skills':[],'raw':copy.deepcopy(state['raw'])};states['FIXED']={'skills':[],'raw':{}}
 write('FROZEN_STATES.json',{'at':time.time(),'states':states,'selection_rule':'final accepted state after fixed candidate budget; no confirmation-based changes'})
 confirmation=json.loads((R/'private_confirmation.json').read_text())
 results={}
 for arm in P['evaluation']['confirmation_arms']:
  results[arm]={}
  for task in P['tasks']:
   results[arm][task]=run_set(task,confirmation,states[arm],f'confirmation_{arm}',fixed=arm=='FIXED')
 write('private_confirmation_results.json',results)
 write('learning_summary.json',{'gates':gates,'validation':val_results,'final_skill_count':len(state['skills']),'vlm_calls':models.calls,'sam_calls':models.sam_calls})
 from analyze import analyze
 analyze(R)
 write('EXIT.json',{'success':True,'ended':time.time(),'vlm_calls':models.calls,'sam_calls':models.sam_calls,'elapsed_seconds':time.time()-B['started_epoch']})
except BaseException:
 write('EXIT.json',{'success':False,'ended':time.time(),'error':traceback.format_exc(),'original_start':B['started_epoch']});raise
