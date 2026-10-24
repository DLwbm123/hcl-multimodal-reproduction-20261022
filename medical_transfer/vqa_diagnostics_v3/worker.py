"""Two bounded development experiments with explicit independent review boundaries."""
import os,sys,json,time,copy,traceback
from pathlib import Path
R=Path(os.environ['RUN_ROOT']);role=os.environ['RUN_ROLE'];O=R/role;O.mkdir(exist_ok=True)
sys.path[:0]=[str(R/'code'),str(R/'source'),str(R/'source/medical_transfer/src'),str(R/'source/medical_transfer/development_diagnostic'),str(R/'source/medical_transfer/template_contract'),str(R/'source/medical_transfer/semantic_boundary')]
P=json.loads((R/'protocol.json').read_text());B=json.loads((R/'budget.json').read_text())
os.environ.update(HF_HOME=str(R/'cache'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TMPDIR=str(R/'tmp'),TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',CUDA_CACHE_PATH=str(R/'cache/cuda'),XDG_CACHE_HOME=str(R/'cache'),TORCH_HOME=str(R/'cache/torch'),TRITON_CACHE_DIR=str(R/'cache/triton'),PYTHONDONTWRITEBYTECODE='1')
import tempfile
tempfile.tempdir=str(R/'tmp')
from adapter import install_stream,snapshot
from local_update import candidates,safe_metrics
from diagnostics import quality_counts,choose_c,choose_gate,memory_change,install_train_retrieval,contract_text,install_contract,shifted_images
from runtime_support import install_protocol
from semantic_boundary import install_semantic_boundary
from medical_hcl.bridge import construct_pipeline
from medical_hcl.evaluation import MedicalEvaluator
from hcl.models import LocalTransformersModel,_state_phase
import torch

def write(name,obj):
 p=O/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2,allow_nan=False));t.replace(p)
def append(name,obj):
 with (O/name).open('a') as f:f.write(json.dumps(obj,allow_nan=False)+'\n')
def guard():
 s=json.loads((R/'resource_status.json').read_text())
 if (R/'HARD_STOP.json').exists() or time.time()-s['time']>120 or time.time()>=B['soft_deadline_epoch']:raise RuntimeError('guard stale or deadline')
class Model(LocalTransformersModel):
 calls=0;scope='loading';expected_image=None
 def generate(self,prompt,*,state):
  guard()
  if _state_phase(state)=='final_answer' and self.expected_image is not None:
   assert str(self._image_path_from_state(state))==str(self.expected_image),'image permutation/input mismatch'
  if self.calls>=P['budget']['max_calls_by_role'][role]:raise RuntimeError('call cap')
  self.calls+=1;write('model_attempts.json',{'attempts':self.calls});start=time.monotonic();ok=False
  try:answer=super().generate(prompt,state=state);ok=True;return answer
  finally:append('call_costs.jsonl',{'attempt':self.calls,'scope':self.scope,'phase':_state_phase(state),'seconds':time.monotonic()-start,'success':ok,'image_supplied':bool(self._image_path_from_state(state)),'peak_allocated_bytes':torch.cuda.max_memory_allocated()})

def build(name,state=None,forced=None,contract=None):
 pipe=construct_pipeline(os.environ['PREPARED_DATA'],O/'state'/name,model=model,real_execution_authorized=True,arm='stability_hcl')
 install_protocol(pipe,model,O,P);install_semantic_boundary(pipe);install_stream(pipe,'stability',P)
 if state is not None:pipe.memory.restore(state)
 if forced is not None:install_train_retrieval(pipe,forced)
 if contract is not None:install_contract(pipe,contract)
 assert not pipe.prediction_cache.enabled and not pipe.optimizer.components
 return pipe

def evalset(pipe,name,examples):
 before=snapshot(pipe);preds=[]
 for i,e in enumerate(examples):
  guard();model.scope='evaluation:'+name;model.expected_image=e['image']
  pred=pipe._predict([e],update_memory=False,progress_label=model.scope)[0];preds.append(pred)
  append('private_predictions.jsonl',{'block':name,'prediction':pred})
  write('progress.json',{'phase':'evaluation','block':name,'completed':i+1,'total':len(examples),'calls':model.calls})
 assert snapshot(pipe)==before and all(not p.requires_grad for p in model._model.parameters()) and not model._model.training
 result=evaluator.evaluate(examples,preds,split=name)
 write('private_eval_'+name+'.json',result)
 append('readonly_checks.jsonl',{'block':name,'n':len(examples),'passed':True})
 selections=[p.get('trace',{}).get('selected_memory',[]) for p in preds]
 summary={'metrics':result['metrics'],'quality':quality_counts(examples,result['case_results']),'retrieval':{'selected_total':sum(map(len,selections)),'empty':sum(not x for x in selections),'context_chars_total':sum(len(str(p.get('trace',{}).get('prompt',''))) for p in preds)}}
 append('evaluation_summaries.jsonl',{'block':name,**summary})
 return result,summary

def proposal(pipe,train,name):
 model.scope='proposal:'+name;model.expected_image=None;before=pipe.memory.snapshot();old=pipe.memory.record_training_interaction;sequence=[]
 def record(task_name,e,prediction):
  old(task_name,e,prediction)
  entry={'task_id':e['task_id'],'prediction':prediction,'raw_after':copy.deepcopy(pipe.memory.raw.records)};sequence.append(entry);append('private_train_interactions.jsonl',{'block':name,**entry})
 pipe.memory.record_training_interaction=record
 try:pipe._evaluate_train_batch(train,split='proposal:'+name)
 finally:pipe.memory.record_training_interaction=old
 state=pipe.memory.snapshot();assert len(sequence)==len(train)
 write('private_proposal_'+name+'.json',{'before':before,'after':state,'train_ids':[e['task_id'] for e in train]});pipe.memory.restore(before)
 return state

def correctness(result):return {x['task_id']:bool(x['success']) for x in result['case_results']}
def change(before,after):
 b=correctness(before);a=correctness(after);assert set(a)==set(b)
 return {'n':len(a),'before_correct':sum(b.values()),'after_correct':sum(a.values()),'gained':sum(a[k] and not b[k] for k in a),'lost':sum(b[k] and not a[k] for k in a)}

def run_p1():
 base=json.loads((R/'private_after_B.json').read_text());train=data['C']['train'][:24];sel=data['C']['selection'];results={};summaries={}
 empty=build('empty');results['empty'],summaries['empty']=evalset(empty,'empty_selection',sel)
 pipe=build('common',base);results['after_B'],summaries['after_B']=evalset(pipe,'after_B_selection',sel)
 prop=proposal(pipe,train,'C');local=dict(candidates(base,prop,train,0,True))['raw_only'];full=dict(candidates(base,prop,train,0,False))['raw_only']
 trainids={e['task_id'] for e in train};forced={x['memory_id'] for x in prop['raw'] if x['internal_source']['task_id'] in trainids};contract=contract_text(train)
 specs={'local_raw':(local,None,None),'bulk_raw':(full,None,None),'bulk_raw_train_only_lexical':(full,forced,None),'bulk_raw_train_only_lexical_contract':(full,forced,contract)}
 for name,(state,ids,text) in specs.items():
  cp=build(name,state,ids,text);results[name],summaries[name]=evalset(cp,name+'_selection',sel)
  write('private_candidate_'+name+'.json',state)
 chosen=choose_c(summaries)
 write('FROZEN_CHOICE.json',{'chosen':chosen,'chosen_before_review_inference':True,'selection_results':summaries,'at':time.time()})
 review=None;promoted=False
 if chosen is not None:
  baseline,bs=evalset(pipe,'after_B_review',data['C']['review']);st,ids,text=specs[chosen]
  cp=build('review_chosen',st,ids,text);candidate,cs=evalset(cp,'chosen_review',data['C']['review'])
  review={'baseline':bs,'chosen':cs,'paired_change':change(baseline,candidate)}
  promoted=cs['quality']['correct']>bs['quality']['correct'] and cs['quality']['flag_free_correct']>bs['quality']['flag_free_correct']
 real,rs=evalset(pipe,'B_true_image',data['B']['selection']);shuffled,ss=evalset(pipe,'B_shifted_image',shifted_images(data['B']['selection']))
 return {'experiment':'P1_C_bottleneck','selection':summaries,'chosen':chosen,'review':review,'promotion':promoted,'original_native_score_preserved':True,'train_only_forced_records_available':len(forced),'memory_change':{'local_raw':memory_change(base,local),'bulk_raw':memory_change(base,full)},'image_dependence_B':{'true':rs,'shifted':ss,'paired_change_true_to_shifted':change(real,shuffled)},'missing_candidate_states':False,'scope':'development diagnosis, not independent method confirmation'}

def run_p2():
 common=json.loads((R/'private_after_A.json').read_text());assert len(common['raw'])==10 and len(common['abstract'])==2
 pipe=build('common',common);empty=build('empty')
 baseline_A,bs=evalset(empty,'empty_A_review',data['A']['review'])
 review=data['A']['review']+data['B']['review'];common_review,common_summary=evalset(pipe,'common_review',review)
 common_map=correctness(common_review);base_map=correctness(baseline_A);new_A={k for k in base_map if not base_map[k] and common_map[k]}
 gate=data['B']['selection']+data['A']['anchor'];reference,_=evalset(pipe,'common_gate',gate)
 def gate_score(result):return pipe._evaluate_gate(gate,result['predictions'],current_val_examples=data['B']['selection'],historical_anchor_examples=data['A']['anchor'],split='paired_gate',reference_predictions=reference['predictions'])
 ref=gate_score(reference);blocks=[]
 for b in range(2):
  pipe.memory.restore(common);train=data['B']['train'][b*24:(b+1)*24];prop=proposal(pipe,train,'B'+str(b+1));families={}
  for family in ['local','bulk']:
   states=candidates(common,prop,train,b,family=='local');scores=[]
   for i,(scope,state) in enumerate(states):
    pipe.memory.restore(state);name=f'block{b+1}_{family}_{scope}';res,summary=evalset(pipe,name+'_gate',gate);scores.append(gate_score(res))
    write('private_candidate_'+name+'.json',state)
   families[family]={'states':states,'scores':scores}
  policies={};review_cache={'common':(common_review,common_summary)}
  for family,vals in families.items():
   for budget,label in [(0,'0'),(None,'inf')]:
    winner,decisions=choose_gate(pipe,ref,vals['scores'],budget);key='common' if winner is None else family+str(winner)
    state=common if winner is None else vals['states'][winner][1]
    if key not in review_cache:
     pipe.memory.restore(state);review_cache[key]=evalset(pipe,f'block{b+1}_{family}{label}_review',review)
    result,summary=review_cache[key];now=correctness(result);bids={e['task_id'] for e in data['B']['review']}
    policies[family+label]={'accepted':winner is not None,'chosen_index':winner,'candidate_decisions':decisions,'candidate_metrics':[safe_metrics(x) for x in vals['scores']],'memory_change':memory_change(common,state),'review_summary':summary,'new_A_reference_count':len(new_A),'new_A_lost':sum(not now[k] for k in new_A),'B_review_gain':sum(int(now[k])-int(common_map[k]) for k in bids)}
  local=policies['local0'];bulk_inf=policies['bulkinf'];bulk0=policies['bulk0']
  blocked=any(x['reasons']==['historical_loss_budget_exceeded'] for x in bulk0['candidate_decisions'])
  passed=bool(new_A) and local['accepted'] and local['B_review_gain']>0 and local['new_A_lost']==0 and bulk_inf['accepted'] and bulk_inf['candidate_metrics'][bulk_inf['chosen_index']]['historical_anchor_metrics'].get('forget_count',0)>0 and not bulk0['accepted'] and blocked
  blocks.append({'block':b+1,'policies':policies,'mechanism_pattern_passed':passed});write('partial_results.json',{'new_A_review_count':len(new_A),'blocks':blocks})
 return {'experiment':'P2_common_learned_A','common_review':common_summary,'empty_A_review':bs,'new_A_review_count':len(new_A),'blocks':blocks,'replicated_mechanism':all(x['mechanism_pattern_passed'] for x in blocks),'scope':'two independent one-step blocks from the same exploration-selected full H_A, not a new continuous sequence'}

try:
 guard();write('START.json',{'pid':os.getpid(),'pgid':os.getpgrp(),'started':time.time(),'original_start':B['started_epoch'],'gpu':os.environ['CUDA_VISIBLE_DEVICES']})
 torch.manual_seed(P['seed'])
 model=Model(name='Qwen2.5-VL-7B-Instruct',path=B['model_path'],torch_dtype='bfloat16',attn_implementation='sdpa',max_new_tokens=2048,answer_max_new_tokens=512,temperature=0.,enable_thinking=False,local_files_only=True)
 model._load();assert not model._model.training and all(not p.requires_grad for p in model._model.parameters())
 write('MODEL_READY.json',{'at':time.time(),'parameters_frozen':True,'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
 data=json.loads((R/'private_data.json').read_text());evaluator=MedicalEvaluator(json.loads((Path(os.environ['PREPARED_DATA'])/'class_vocabulary.json').read_text()))
 result=run_p1() if role=='p1' else run_p2();write('RESULT.json',result)
 write('EXIT.json',{'exit_code':0,'ended':time.time(),'elapsed_seconds':time.time()-B['started_epoch'],'calls':model.calls,'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
except BaseException:
 (O/'FAILURE.txt').write_text(traceback.format_exc());write('EXIT.json',{'exit_code':1,'ended':time.time()});raise
