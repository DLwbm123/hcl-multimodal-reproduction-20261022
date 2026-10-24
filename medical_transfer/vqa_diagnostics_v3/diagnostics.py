"""Pure decision helpers and declared diagnostic interventions."""
import copy,json,re
from types import MethodType

def flagged(e):
 return str(e['question']).strip().casefold() in {'','nan','null','none'} or str(e['answer']).strip().casefold() in {'','nan','null'}

def quality_counts(examples,rows):
 by={x['task_id']:x for x in rows};valid=[e for e in examples if not flagged(e)]
 return {'n':len(examples),'correct':sum(x['success'] for x in rows),'flagged':len(examples)-len(valid),'flag_free_n':len(valid),'flag_free_correct':sum(by[e['task_id']]['success'] for e in valid)}

def choose_c(results):
 order=['local_raw','bulk_raw','bulk_raw_train_only_lexical','bulk_raw_train_only_lexical_contract']
 base=results['after_B'];eligible=[]
 for i,name in enumerate(order):
  r=results[name]
  if r['quality']['correct']>=base['quality']['correct']+2 and r['quality']['flag_free_correct']>=base['quality']['flag_free_correct']+2 and r['metrics']['format_compliance_rate']==1:
   eligible.append((r['quality']['correct'],-i,name))
 return max(eligible)[-1] if eligible else None

def choose_gate(pipe,reference,results,budget):
 previous=pipe.optimizer.historical_loss_budget;pipe.optimizer.historical_loss_budget=budget
 try:
  eligible=[];decisions=[]
  for i,r in enumerate(results):
   yes,reasons=pipe._memory_candidate_decision(reference,r)
   decisions.append({'eligible':yes,'reasons':reasons})
   if yes:eligible.append((r['current_task_metrics']['correct'],-r['historical_anchor_metrics'].get('forget_count',0),-i,i))
  return (max(eligible)[-1] if eligible else None),decisions
 finally:pipe.optimizer.historical_loss_budget=previous

def memory_change(before,after):
 out={}
 for bank in ['raw','abstract']:
  old={x['memory_id']:x for x in before[bank]};new={x['memory_id']:x for x in after[bank]}
  out[bank]={'before':len(old),'after':len(new),'unchanged':sum(k in new and new[k]==v for k,v in old.items()),'removed_ids':len(set(old)-set(new)),'new_ids':len(set(new)-set(old)),'modified_existing':sum(k in old and old[k]!=v for k,v in new.items())}
 return out

def install_train_retrieval(pipe,allowed_ids):
 def select(self,*,current_input,workflow_decision,**kwargs):
  tok=lambda x:set(re.findall(r'[a-z0-9]+',x.lower()));query=tok(json.dumps(current_input,sort_keys=True))
  pool=[x for x in self.retrieval_candidates(current_input=current_input,workflow_decision=workflow_decision) if x.get('memory_id') in allowed_ids]
  def rank(x):
   words=tok(x.get('question',''));u=query|words
   return (-(len(query&words)/len(u) if u else 0),str(x['memory_id']))
  selected=[{**x,'selected_role':'support','selection_reason':'train-only lexical diagnostic','selection_risk':'medium'} for x in sorted(pool,key=rank)[:2]]
  return selected,{'selected':[{'memory_id':x['memory_id']} for x in selected],'candidate_count':len(pool),'policy':'train_only_lexical_diagnostic'}
 pipe.memory.select_for_context=MethodType(select,pipe.memory)

def contract_text(train):
 regions=sorted({token.strip() for e in train if e['metadata']['question_type']=='location_set' and not flagged(e) for token in e['answer'].split(';')})
 assert regions and all(len(x)<40 for x in regions)
 return ('Diagnostic answer contract (vocabulary learned only from training examples): Use the current image to answer the question. '
 'Output only the shortest answer, without explanations. If the requested object or finding is absent, output none. '
 'For image location questions, use the applicable labels from this training vocabulary: '+', '.join(regions)+'. '
 'If multiple locations apply, separate them with semicolons. Do not infer presence from this vocabulary or historical examples.')

def install_contract(pipe,text):
 original=pipe._predict_with_router
 def predict(self,examples,router,**kwargs):
  old=router.build_context
  def build(router_self,*args,**kw):return old(*args,**kw)+'\n\n'+text
  router.build_context=MethodType(build,router)
  try:return original(examples,router,**kwargs)
  finally:router.build_context=old
 pipe._predict_with_router=MethodType(predict,pipe)

def shifted_images(examples):
 result=copy.deepcopy(examples);images=[e['image'] for e in examples];assert len(images)>1 and len(set(images))==len(images)
 for i,e in enumerate(result):e['image']=images[(i+1)%len(images)]
 return result
