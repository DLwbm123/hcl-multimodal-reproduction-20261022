"""Evaluate persistent training experience in a post-draft verification workflow."""
import json,os,signal,subprocess,threading,time,traceback
from pathlib import Path
import run as core
from logic import paired
from storage import tree_bytes
from review_logic import ARMS,FINAL,review_text,parse_option,decision
ROOT=core.ROOT;BUDGET=core.BUDGET
INPUTS=[];STATES={};FATAL=None
os.environ['TMPDIR']='/data/bmw/workers/t1014';os.environ['TRITON_CACHE_DIR']='/data/bmw/workers/t1014/triton'


def watchdog():
    roots=[ROOT]+[ROOT.parent/n for n in ['medrax_subscription_v4','medrax_hcl_v5','medrax_memory_interface_v6','medrax_corrective_memory_v7','medrax_error_abstraction_v8','medrax_learning_qualification_v9','medrax_evidence_memory_v10','medrax_visual_memory_v11','medrax_autonomy_12h_20261025']]
    while True:
        try:
            sizes=[tree_bytes(p) for p in roots];core.save('RUN_GUARD.json',{'time':time.time(),'own_bytes':sizes[0],'aggregate_bytes':sum(sizes),'deadline':BUDGET['hard_deadline_epoch']})
            if sizes[0]>=BUDGET['own_stop_new_bytes'] or sum(sizes)>=BUDGET['aggregate_stop_new_bytes'] or time.time()>=BUDGET['hard_deadline_epoch']:raise RuntimeError('Original time or storage stop')
        except BaseException as exc:
            try:core.save('STOPPED.json',{'time':time.time(),'error':repr(exc),'context':core.CONTEXT})
            finally:os.killpg(os.getpgrp(),signal.SIGTERM)
            return
        time.sleep(15)


def episode(item,arm,block,tools,tool_state,Agent,prompt,memory):
    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.memory import MemorySaver
    core.guard();label=arm+'_r'+str(block);core.CONTEXT.update(phase='case',episode=len(core.EPISODES),stage='A',state=label)
    memory.restore(STATES[arm]);before=memory.snapshot();paths=item['images'];tool_state.update(calls=0,success=0,images=set(paths))
    model=core.rt.SubscriptionModel();model.deadline=min(time.time()+core.PLAN['limits']['per_case_seconds'],BUDGET['hard_deadline_epoch'])
    outdir=ROOT/'outputs'/str(len(core.EPISODES));outdir.mkdir(exist_ok=False)
    agent=Agent(model,tools,system_prompt=prompt,checkpointer=MemorySaver(),log_dir=str(outdir));config={'configurable':{'thread_id':'1'},'recursion_limit':2*core.PLAN['limits']['per_case_controller_calls']+2}
    figure=''.join(Path(p).name+' located at '+p+'\n' for p in paths)
    user='Answer this question correctly using chain of thought reasoning and carefully evaluating choices. Solve using our own vision and reasoning and thenuse tools to complement your reasoning. Trust your own judgement over any tools.\n'+item['question']+'\n'+figure
    content=[{'type':'text','text':user}]+[{'type':'image_url','image_url':{'url':p}} for p in paths]
    text=review_text(arm,STATES);result={'id':item['id'],'stage':'A','role':'adaptive_error_enriched_development','state':label,'status':'failed','predicted':None,'initial_predicted':None,'review_predicted':None,'review_entered':False,'review_context_chars':len(text),'memory_ids':[r['memory_id'] for r in STATES[arm]['abstract']]};started=time.time()
    try:
        out=agent.workflow.invoke({'messages':[HumanMessage(content=content)]},config);initial=out['messages'][-1].content;result.update(initial_final=initial,initial_predicted=parse_option(initial))
        if text:
            result['review_entered']=True;out=agent.workflow.invoke({'messages':[HumanMessage(content=text)]},config);reviewed=out['messages'][-1].content;result.update(review_final=reviewed,review_predicted=parse_option(reviewed))
        out=agent.workflow.invoke({'messages':[HumanMessage(content=FINAL)]},config);answer=out['messages'][-1].content.strip();result.update(status='completed',final=answer,predicted=answer if answer in list('ABCDEF') else None)
    except Exception as exc:result['error']=repr(exc);core.record('episode_error',{'error':repr(exc)})
    assert memory.snapshot()==before
    result.update(seconds=time.time()-started,controller_calls=model.calls,tool_calls=tool_state['calls'],successful_tools=tool_state['success']);core.EPISODES.append(result);core.save('private/episodes.json',core.EPISODES);core.save('PROGRESS.json',{'time':time.time(),'episodes':len(core.EPISODES),'context':core.CONTEXT,'last_status':result['status']});core.guard()


def finalize():
    gold=json.loads((ROOT/'private/sealed_development_gold.json').read_text());ids=[x['id'] for x in INPUTS];by={};metrics={};pairs=[]
    for block in [1,2]:
        for arm in ARMS:
            label=arm+'_r'+str(block);actual={r['id']:r for r in core.EPISODES if r['state']==label};by[label]={k:dict(actual.get(k,{'id':k,'status':'not_run','predicted':None})) for k in ids}
            metrics[label]={'denominator':len(ids),'correct':sum(r.get('predicted')==gold[k] for k,r in by[label].items()),'valid':sum(r.get('status')=='completed' and r.get('predicted') in list('ABCDEF') for r in actual.values()),'initial_parseable':sum(r.get('initial_predicted') in list('ABCDEF') for r in actual.values()),'initial_correct':sum(r.get('initial_predicted')==gold[k] for k,r in actual.items()),'review_entered':sum(r['review_entered'] for r in actual.values()),'review_parseable':sum(r.get('review_predicted') in list('ABCDEF') for r in actual.values()),'within_run_initial_wrong_to_final_correct':sum(r.get('initial_predicted') in list('ABCDEF') and r['initial_predicted']!=gold[k] and r.get('predicted')==gold[k] for k,r in actual.items()),'within_run_initial_correct_to_final_wrong':sum(r.get('initial_predicted')==gold[k] and r.get('predicted')!=gold[k] for k,r in actual.items())}
        get=lambda a:list(by[a+'_r'+str(block)].values());pairs.append({k:paired(get(a),get('experience_review'),ids,gold) for k,a in [('H0_to_experience','H0_static'),('generic_to_experience','generic_review')]})
    complete=FATAL is None and len(core.EPISODES)==36 and all(m['valid']==6 for m in metrics.values());gate=decision(pairs,by,ids,gold,complete);core.save('private/gate.json',gate);gate={k:v for k,v in gate.items() if k!='repeat_observed_attributable_repair_ids'}
    report={'time':time.time(),'fatal':FATAL,'complete':complete,'planned_episodes':36,'physical_episodes':len(core.EPISODES),'metrics':metrics,'paired_per_repeat':pairs,'acquisition_gate':gate,'error_enriched_reused_development_cases':6,'confirmation_used':False,'new_memory_generation_calls':0,'medical_validity_established':False,'sequential_learning_established':False,'seconds_since_original_start':time.time()-BUDGET['started_epoch']};core.save('private/results.json',by);core.save('AGGREGATE_REPORT.json',report);core.save('FINISHED.json',report)


def main():
    global INPUTS,STATES,FATAL
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('Independent watchdog termination')));threading.Thread(target=watchdog,daemon=True).start();core.save('STARTED.json',{'time':time.time(),'original_start':BUDGET['started_epoch'],'deadline':BUDGET['hard_deadline_epoch']})
    INPUTS=json.loads((ROOT/'private/inputs.json').read_text());STATES=json.loads((ROOT/'private/states.json').read_text())
    from hcl.memory import ExperienceMemory,MemoryConfig
    memory=ExperienceMemory(MemoryConfig(record_dir=str(ROOT/'private/memory'),raw_capacity_per_task=24,abstract_capacity_per_task=2,skill_enabled=False,anchor_selection_mode='off'))
    try:
        s=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True);free={int(x.split(',')[0]):int(x.split(',')[1]) for x in s.strip().splitlines()};gpu=int(os.environ['CUDA_VISIBLE_DEVICES']);core.save('GPU_PREFLIGHT.json',{'time':time.time(),'gpu':gpu,'free_mib':free[gpu]})
        if free[gpu]<45000:raise RuntimeError('Insufficient GPU headroom')
        tools,state,Agent,prompt=core.rt.load_six();core.save('TOOLS_LOADED.json',{'time':time.time(),'names':[t.name for t in tools]})
        for block in [1,2]:
            for i,item in enumerate(INPUTS if block==1 else reversed(INPUTS)):
                offset=(i+block-1)%3
                for arm in ARMS[offset:]+ARMS[:offset]:episode(item,arm,block,tools,state,Agent,prompt,memory)
        assert STATES==json.loads((ROOT/'private/states.json').read_text())
    except BaseException as exc:FATAL=repr(exc);core.save('FAILED.json',{'time':time.time(),'error':FATAL,'traceback':traceback.format_exc(),'context':core.CONTEXT})
    finally:finalize()
    print(json.dumps({'finished':True,'fatal':FATAL,'episodes':len(core.EPISODES)}),flush=True)

if __name__=='__main__':main()
