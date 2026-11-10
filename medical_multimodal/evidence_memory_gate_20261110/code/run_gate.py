"""Native MedRAX evaluates a frozen HCL candidate without state updates."""
import json,os,signal,time,traceback
from pathlib import Path
from copy import deepcopy
import base_worker as base
import build_candidate as generation
from evidence_memory import EvidenceMemory
from hcl.memory import MemoryConfig
from hcl.router import Router,RouterConfig
from hcl.task_interface import TaskInterface,TaskInterfaceConfig
from local_controller import LocalController
from qualified_tool import install_shared_guard
from policy import FIXED
from gate import evaluate
ROOT,PLAN=base.ROOT,base.PLAN
EXTRA=''
base.policy_context=lambda *_:(EXTRA,[])
generation.guard=base.guard

def route(candidate,item,memory,model):
    memory.restore(candidate['memory'])
    interface=TaskInterface(TaskInterfaceConfig(mode='rule'))
    chunk=interface.build_taskinterfacechunk({'task_id':item['id'],'question':item['question'],'image_paths':item['images'],
        'visible_context':{'image_count':len(item['images']),'tool_eligible_count':len(item['tool_eligible_images'])}},0)
    router=Router(model,RouterConfig(use_llm_workflow_selector=False,use_tool_selector=False,use_memory_selector=True,use_skill_selector=True,
        memory_selector_invalid_output_retries=0,memory_selector_fallback_to_empty_on_invalid=False,skill_selector_invalid_output_retries=0,skill_selector_fallback_to_empty_on_invalid=False),memory=memory,
        memory_selector_template_content=candidate['memory_selector'],context_template_content={'template_id':'medical.operation.evidence.v1','sections':[
          'Historical operation guidance is fallible; it is not evidence about this patient. Only images and clinical context explicitly provided in the current question are available. Do not invent additional tests or history.',
          'Raw feedback contains paired execution observations and possibly unknown benchmark outcomes. Formatting or empty-observation gains do not establish tool usefulness.',
          'Current task: {model_visible_input}','Workflow: {workflow_decision}','Selected memory: {selected_memory}','Selected skills: {selected_skills}',
          'Use current image/call-bound observations only when relevant and sufficiently informative. The native Agent chooses real tools; keep all shared limits. {selected_tools}']})
    skills=router.select_skills(chunk);workflow=router.plan_workflow(chunk,selected_skills=skills);selected,trace=router.select_memory(chunk,workflow)
    context=router.build_context(chunk,workflow_decision=workflow,selected_memory=selected,selected_skills=skills,selected_tools=[])
    base.record('hcl_route',{'selection':trace,'selected_memory':selected,'selected_skills':skills})
    return FIXED+'\n'+context

def main():
    global EXTRA
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('Independent supervisor stop')))
    inputs=json.loads((ROOT/'private/gate_inputs.json').read_text());candidate=json.loads((ROOT/'private/candidate_memory.json').read_text())
    assert len(inputs)==24 and len({x['source_key'] for x in inputs})==24 and not candidate['accepted']
    assert all(x['privacy_approved'] and all(Path(p).is_file() for p in x['images']) for x in inputs)
    frozen=deepcopy(candidate);fatal=None
    base.save('GATE_STARTED.json',{'time':time.time(),'original_start':base.BUDGET['started_epoch'],'planned_episodes':48})
    try:
        model=generation.Controller();model.batch=[]
        LocalController.model=model.model
        from transformers import AutoProcessor
        LocalController.processor=AutoProcessor.from_pretrained(os.environ['LOCAL_MODEL_PATH'],local_files_only=True,max_pixels=PLAN['controller']['max_pixels'])
        memory=EvidenceMemory(MemoryConfig(record_dir=str(ROOT/'private/gate_transient_memory'),raw_capacity_per_task=46,abstract_mode='llm',anchor_selection_mode='off'),model=model)
        empty=memory.snapshot()
        tools,state,Agent,prompt=base.load_four();install_shared_guard(tools,state,base.record)
        base.save('GATE_TOOLS_LOADED.json',{'time':time.time(),'names':[t.name for t in tools]})
        for i,source in enumerate(inputs):
            for arm in (['Fixed','memory'] if i%2==0 else ['memory','Fixed']):
                item=dict(source);started=time.time();item['_case_started']=started
                base.CONTEXT.update(phase='gate',episode_id=item['id'],arm=arm,case_deadline=started+900)
                before=len(generation.CALLS);count=len(base.EPISODES)
                try:
                    memory.restore(empty if arm=='Fixed' else candidate['memory'])
                    EXTRA=FIXED if arm=='Fixed' else route(candidate,item,memory,model)
                    if len(generation.CALLS)-before>2:raise RuntimeError('Routing call budget exceeded')
                    base.episode(item,arm,tools,state,Agent,prompt,None,None)
                except Exception as exc:
                    if len(base.EPISODES)!=count:raise
                    base.EPISODES.append({'id':item['id'],'arm':arm,'status':'failed','predicted':None,'error':repr(exc),'tool_calls':0,'successful_tools':0,'controller_calls':0})
                    base.record('route_failure',{'error':repr(exc)})
                finally:
                    signal.setitimer(signal.ITIMER_REAL,0)
                result=base.EPISODES[-1];result.update(gate_role=item['gate_role'],seconds=time.time()-started,route_calls=len(generation.CALLS)-before)
                if candidate!=frozen or memory.snapshot()!=(empty if arm=='Fixed' else candidate['memory']):raise RuntimeError('Evaluation mutated frozen candidate')
                base.save('private/episodes.json',base.EPISODES)
                base.save('PROGRESS.json',{'time':time.time(),'recorded':len(base.EPISODES),'planned':48,'last_status':result['status']})
        memory.restore(empty);assert memory.snapshot()==empty
    except BaseException as exc:
        fatal=repr(exc);base.save('FAILED.json',{'error':fatal,'traceback':traceback.format_exc()})
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);base.CONTEXT.pop('case_deadline',None)
        targets=json.loads((ROOT/'private/gate_targets.json').read_text())
        report=evaluate(inputs,base.EPISODES,targets);report.update(fatal=fatal,elapsed_since_original_start=time.time()-base.BUDGET['started_epoch'],recorded=len(base.EPISODES),planned=48)
        if fatal:report['accepted_for_research_development']=False
        base.save('GATE_REPORT.json',report);base.save('GATE_FINISHED.json',report)
        base.save('private/ACCEPTANCE.json',{'accepted':report['accepted_for_research_development'],'candidate':'memory' if report['accepted_for_research_development'] else None,'prior_restored':not report['accepted_for_research_development'],'clinical_deployment':False})
    return 2 if fatal else 0

if __name__=='__main__':raise SystemExit(main())
