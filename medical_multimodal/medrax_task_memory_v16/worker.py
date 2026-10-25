"""Actual task-B raw memory update with matched A-only state and a small A retention panel."""
import json,os,signal,subprocess,threading,time,traceback
from pathlib import Path
import run as core
from logic import paired
from storage import tree_bytes
from task_memory import ARMS,FINAL,select,context,gate
ROOT=core.ROOT;BUDGET=core.BUDGET
INPUTS=[];STATES={};SUPPORT={};FATAL=None
os.environ['TMPDIR']='/data/bmw/workers/t1024';os.environ['TRITON_CACHE_DIR']='/data/bmw/workers/t1024/triton'

def watchdog():
    roots=[ROOT]+[ROOT.parent/n for n in ['medrax_subscription_v4','medrax_hcl_v5','medrax_memory_interface_v6','medrax_corrective_memory_v7','medrax_error_abstraction_v8','medrax_learning_qualification_v9','medrax_evidence_memory_v10','medrax_visual_memory_v11','medrax_postdraft_memory_v12','medrax_tool_query_v13','medrax_joint_query_v14','medrax_visual_focus_v15','medrax_autonomy_12h_20261025']]
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
    core.guard();stage=item['stage'];label=stage+'_'+arm+'_r'+str(block);core.CONTEXT.update(phase='case',episode=len(core.EPISODES),stage=stage,state=label)
    snapshot=STATES[arm];memory.restore(snapshot);before=memory.snapshot();paths=item['images'];tool_state.update(calls=0,success=0,images=set(paths));records=select(before,item['question'],stage,arm);retrieved=[r['memory_id'] for r in records];assert SUPPORT[item['id']][arm]==retrieved;text=context(records)
    model=core.rt.SubscriptionModel();model.deadline=min(time.time()+core.PLAN['limits']['per_case_seconds'],BUDGET['hard_deadline_epoch'])
    outdir=ROOT/'outputs'/str(len(core.EPISODES));outdir.mkdir(exist_ok=False)
    agent=Agent(model,tools,system_prompt=prompt,checkpointer=MemorySaver(),log_dir=str(outdir))
    config={'configurable':{'thread_id':'1'},'recursion_limit':2*core.PLAN['limits']['per_case_controller_calls']+2}
    figure=''.join(Path(p).name+' located at '+p+'\n' for p in paths)
    user='Answer this question correctly using chain of thought reasoning and carefully evaluating choices. Solve using our own vision and reasoning and thenuse tools to complement your reasoning. Trust your own judgement over any tools.\n'+item['question']+'\n'+figure+text
    content=[{'type':'text','text':user}]+[{'type':'image_url','image_url':{'url':p}} for p in paths]
    result={'id':item['id'],'stage':stage,'role':'adaptive_reused_development','state':label,'status':'failed','predicted':None,'retrieved_ids':retrieved,'context_text':text};started=time.time()
    try:
        out=agent.workflow.invoke({'messages':[HumanMessage(content=content)]},config);result['initial_final']=out['messages'][-1].content
        out=agent.workflow.invoke({'messages':[HumanMessage(content=FINAL)]},config)
        answer=out['messages'][-1].content.strip();result.update(status='completed',final=answer,predicted=answer if answer in list('ABCDEF') else None)
    except Exception as exc:result['error']=repr(exc);core.record('episode_error',{'error':repr(exc)})
    assert memory.snapshot()==before
    result.update(seconds=time.time()-started,controller_calls=model.calls,tool_calls=tool_state['calls'],successful_tools=tool_state['success'],context_chars=len(text));core.EPISODES.append(result)
    core.save('private/episodes.json',core.EPISODES);core.save('PROGRESS.json',{'time':time.time(),'episodes':len(core.EPISODES),'context':core.CONTEXT,'last_status':result['status']});core.guard()

def finalize():
    gold=json.loads((ROOT/'private/sealed_development_gold.json').read_text());ids={s:[x['id'] for x in INPUTS if x['stage']==s] for s in ['A','B']};by={};metrics={};pairs=[]
    for b in [1,2]:
        for stage,arms in [('B',ARMS),('A',['H_A','H_B'])]:
            for arm in arms:
                label=stage+'_'+arm+'_r'+str(b);actual={r['id']:r for r in core.EPISODES if r['state']==label};by[label]={k:dict(actual.get(k,{'id':k,'status':'not_run','predicted':None})) for k in ids[stage]};metrics[label]={'denominator':len(ids[stage]),'correct':sum(r.get('predicted')==gold[k] for k,r in by[label].items()),'valid':sum(r['status']=='completed' and r.get('predicted') in list('ABCDEF') for r in actual.values()),'memory_used_episodes':sum(bool(r['retrieved_ids']) for r in actual.values()),'mean_context_chars':sum(r['context_chars'] for r in actual.values())/len(actual) if actual else None}
        get=lambda s,a:list(by[s+'_'+a+'_r'+str(b)].values());pairs.append({'B_H0_to_HB':paired(get('B','H0_static'),get('B','H_B'),ids['B'],gold),'B_HA_to_HB':paired(get('B','H_A'),get('B','H_B'),ids['B'],gold),'A_HA_to_HB':paired(get('A','H_A'),get('A','H_B'),ids['A'],gold)})
    complete=FATAL is None and len(core.EPISODES)==48 and all(v['valid']==v['denominator'] for v in metrics.values());g=gate(pairs,by,ids,gold,complete);core.save('private/gate.json',g);g={k:v for k,v in g.items() if k!='repeated_gain_ids'}
    report={'time':time.time(),'fatal':FATAL,'complete':complete,'planned_episodes':48,'physical_episodes':len(core.EPISODES),'metrics':metrics,'paired_per_repeat':pairs,'acquisition_and_small_retention_gate':g,'reused_B_cases':6,'reused_A_cases':3,'confirmation_used':False,'generation_calls':0,'medical_validity_established':False,'parameter_retention_established':False,'seconds_since_original_start':time.time()-BUDGET['started_epoch']};core.save('private/results.json',by);core.save('AGGREGATE_REPORT.json',report);core.save('FINISHED.json',report)


def main():
    global INPUTS,STATES,SUPPORT,FATAL
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('Independent watchdog termination')));threading.Thread(target=watchdog,daemon=True).start()
    core.save('STARTED.json',{'time':time.time(),'original_start':BUDGET['started_epoch'],'deadline':BUDGET['hard_deadline_epoch']})
    INPUTS=json.loads((ROOT/'private/inputs.json').read_text());STATES=json.loads((ROOT/'private/states.json').read_text());SUPPORT=json.loads((ROOT/'private/support_selection.json').read_text())
    from hcl.memory import ExperienceMemory,MemoryConfig
    memory=ExperienceMemory(MemoryConfig(record_dir=str(ROOT/'private/memory'),raw_capacity_per_task=24,abstract_capacity_per_task=2,skill_enabled=False,anchor_selection_mode='off'))
    try:
        free=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True);available={int(x.split(',')[0]):int(x.split(',')[1]) for x in free.strip().splitlines()};gpu=int(os.environ['CUDA_VISIBLE_DEVICES']);core.save('GPU_PREFLIGHT.json',{'time':time.time(),'gpu':gpu,'free_mib':available[gpu]})
        if available[gpu]<45000:raise RuntimeError('Insufficient GPU headroom')
        core.CONTEXT['phase']='loading_six_tools';tools,state,Agent,prompt=core.rt.load_six();core.save('TOOLS_LOADED.json',{'time':time.time(),'names':[t.name for t in tools]})
        for block in [1,2]:
            for stage,arms in [('B',ARMS),('A',['H_A','H_B'])]:
                items=[x for x in INPUTS if x['stage']==stage]
                for i,item in enumerate(items if block==1 else reversed(items)):
                    offset=(i+block-1)%len(arms)
                    for arm in arms[offset:]+arms[:offset]:episode(item,arm,block,tools,state,Agent,prompt,memory)
        assert STATES==json.loads((ROOT/'private/states.json').read_text())
    except BaseException as exc:FATAL=repr(exc);core.save('FAILED.json',{'time':time.time(),'error':FATAL,'traceback':traceback.format_exc(),'context':core.CONTEXT})
    finally:finalize()
    print(json.dumps({'finished':True,'fatal':FATAL,'episodes':len(core.EPISODES)}),flush=True)

if __name__=='__main__':main()
