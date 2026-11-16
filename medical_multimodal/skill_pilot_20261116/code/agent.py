"""LangGraph controller plus executable skill calls, all under shared primitive budgets."""
import copy,json,time
from typing import TypedDict
from langgraph.graph import StateGraph,END
from core import CATALOG,TASKS,decode,decode_final,resolve,condition,validate_library

class State(TypedDict):
 actions:int
 observations:list
 pending:dict
 answer:dict
 done:bool
 error:str

class SkillExecutor:
 def __init__(self,tools):self.tools=tools;self.branch_events=[]
 def run(self,skill):
  validate_library([skill]);env={'input':{'full_bbox':[0,0,self.tools.w,self.tools.h],'width':self.tools.w,'height':self.tools.h},'last':{'bbox':[0,0,self.tools.w,self.tools.h]}}
  for step in skill['steps']:
   execute=condition(step.get('when'),env)
   if 'when' in step:self.branch_events.append({'step':step['id'],'condition':step['when'],'executed':execute})
   if execute:
    args=resolve(step['args'],env);result=self.tools.execute(step['tool'],args,source=skill['name'])
    env[step['id']]=result
    if result.get('status')!='ok':return {'status':'error','skill':skill['name'],'failed_step':step['id'],'detail':result}
    env['last']=result
   else:env[step['id']]=copy.deepcopy(env['last'])
  return {'status':'ok','skill':skill['name'],'proposed_answer':resolve(skill['output'],env)}

class GraphAgent:
 def __init__(self,models,limits):self.models=models;self.limits=limits
 def run(self,task,tools,library,raw_memory):
  started=time.monotonic();actions=[];uses=[];executor=SkillExecutor(tools)
  visible_skills=validate_library(library)
  instructions=(TASKS[task]+'\nImage size: '+str([tools.w,tools.h])+'.\n'
   'Choose ONE action per turn. Tool action: {"tool":"name","args":{...}}. '
   'Skill action: {"skill":"registered_skill_name"}. Finish action: {"finish":true}; a separate finalizer returns the task answer. '
   'Return exactly one JSON object. Tool and skill outputs can be wrong. Decide from current evidence. '
   'Do not repeat an identical call without new evidence. Stop when sufficient. '
   'All bbox tool inputs and final outputs are ORIGINAL pixels; tools perform their own internal coordinate conversion. '
   'A successful skill returns a proposed_answer, after which you may finish or verify. '
   'The task answer format above is for the finalizer, NEVER your action format. '
   'The localize tool can use args={} to examine the full image without you guessing an ROI. '
   'No need to invoke a skill whose applies_when condition is not met.\nREGISTERED TOOLS (use exact name):\n'+json.dumps([{'name':k,'description':v} for k,v in CATALOG.items()])+'\nSKILLS:\n'+json.dumps(visible_skills)+'\nHISTORICAL TRAINING EXAMPLES (different images; never copy coordinates/mask IDs):\n'+json.dumps(raw_memory))
  def process(s):
   if s['actions']>=self.limits['max_actions'] or self.models.calls-tools.initial_calls>=self.limits['max_vlm_calls_per_episode']-1:
    return {'done':True,'error':''}
   prompt=instructions+'\nOBSERVATIONS:\n'+json.dumps(s['observations'])+'\nRemaining primitive calls: '+str(self.limits['max_primitive_calls']-len(tools.events))
   if s['error']:prompt+='\nPrevious action error: '+s['error']
   raw=self.models.generate(prompt,tools.path,self.limits['controller_max_tokens'],role='controller');actions.append({'raw':raw})
   try:
    a=decode(raw)
    if set(a)=={'action'} and isinstance(a['action'],dict):a=a['action']
    if a=={'tool':'finish','args':{}}:a={'finish':True}
    if set(a)=={'finish'}:
     if a['finish'] is not True:raise ValueError('finish must be true; a separate node returns the answer')
     return {'actions':s['actions']+1,'done':True,'error':''}
    if set(a)=={'skill'}:
     if not isinstance(a['skill'],str) or a['skill'] not in {x['name'] for x in visible_skills}:raise ValueError('unavailable skill')
    elif set(a)=={'tool','args'}:
     if not isinstance(a['tool'],str) or a['tool'] not in CATALOG or not isinstance(a['args'],dict):raise ValueError('Tool must be one of '+', '.join(CATALOG)+'; your invalid action was '+raw[:300])
    else:raise ValueError('action must be tool/args, skill, or finish')
    return {'actions':s['actions']+1,'pending':a,'error':''}
   except (ValueError,TypeError,KeyError) as e:
    return {'actions':s['actions']+1,'pending':{},'error':str(e)[:250]}
  def execute(s):
   a=s['pending']
   try:
    if 'skill' in a:
     uses.append(a['skill']);sk=next(x for x in visible_skills if x['name']==a['skill']);result=executor.run(sk)
    else:result=tools.execute(a['tool'],a['args'])
   except (ValueError,TypeError,KeyError) as e:result={'status':'error','error':str(e)[:250]}
   obs=s['observations']+[{'action':a,'result':result}]
   return {'observations':obs,'pending':{}}
  def finalize(s):
   if not s['observations']:return {'answer':{},'error':'no tool evidence'}
   if self.models.calls-tools.initial_calls>=self.limits['max_vlm_calls_per_episode']:return {'answer':{},'error':'VLM budget exhausted'}
   request=(TASKS[task]+'\nReturn only the task JSON answer, NOT a tool action. '
     'Use the actual CURRENT tool or skill results below. Coordinates and area fractions in those results refer to the ORIGINAL full image. '
     'For T1 select a supported bbox from the results. For T2 select an actual mask_id and its measured area_fraction. '
     'Never copy historical answers.\nCURRENT TOOL RESULTS:\n'+json.dumps(s['observations']))
   raw=self.models.generate(request,tools.path,256,role='finalizer');actions.append({'finalizer_raw':raw})
   try:return {'answer':decode_final(raw,task),'error':''}
   except (ValueError,TypeError) as e:return {'answer':{},'error':'invalid final JSON: '+str(e)}
  builder=StateGraph(State);builder.add_node('process',process);builder.add_node('execute',execute);builder.add_node('finalize',finalize)
  builder.set_entry_point('process');builder.add_conditional_edges('process',lambda s:'end' if s['done'] else 'execute' if s['pending'] else 'retry',{'end':'finalize','execute':'execute','retry':'process'});builder.add_edge('execute','process');builder.add_edge('finalize',END)
  graph=builder.compile();s=graph.invoke({'actions':0,'observations':[],'pending':{},'answer':{},'done':False,'error':''},{'recursion_limit':self.limits['max_actions']*2+3})
  return {'answer':s['answer'],'error':s['error'],'actions':actions,'observations':s['observations'],'tools':tools.events,'skills_used':uses,'branches':executor.branch_events,'vlm_calls':self.models.calls-tools.initial_calls,'sam_calls':tools.sams,'seconds':time.monotonic()-started}

def fixed_run(task,tools):
 start=time.monotonic();a=tools.execute('localize',{})
 try:
  if a.get('status')!='ok':raise ValueError('localization failed')
  if task=='T1':
   b=tools.execute('expand_box',{'bbox':a['bbox'],'fraction':.15})
   c=tools.execute('localize',{'roi':b['bbox']})
   if c.get('status')!='ok':raise ValueError('local refinement failed')
   answer={'bbox':c['bbox']}
  else:
   b=tools.execute('segment',{'bbox':a['bbox']})
   c=tools.execute('measure',{'mask_id':b['mask_id']});answer={'mask_id':c['mask_id'],'area_fraction':c['area_fraction']}
  error=''
 except (ValueError,KeyError,TypeError) as e:answer={};error=str(e)
 return {'answer':answer,'error':error,'actions':[],'observations':[],'tools':tools.events,'skills_used':[],'branches':[],'vlm_calls':tools.models.calls-tools.initial_calls,'sam_calls':tools.sams,'seconds':time.monotonic()-start}
