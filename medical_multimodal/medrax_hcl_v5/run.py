"""Frozen two-stage memory-only experiment using the qualified native agent."""
import copy
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import traceback

import run_agent as rt
from local_update import local_bank
from logic import gate,paired

ROOT=rt.ROOT;PLAN=rt.PLAN;BUDGET=rt.BUDGET
BASE=Path(os.environ['R996_BASE'])
os.environ['TMPDIR']='/data/bmw/workers/t996';os.environ['TRITON_CACHE_DIR']='/data/bmw/workers/t996/triton'
CONTEXT={'phase':'initializing','episode':None,'stage':None,'state':None}
EMPTY={'raw':[],'abstract':[],'anchor':[],'skills':[]}
STATES={'H0_static':copy.deepcopy(EMPTY)};EPISODES=[];INPUTS=[];DECISIONS={};STATE_IDS={};FATAL=None


def save(name,value):
    p=ROOT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2,default=str));t.replace(p)


def record(kind,value):
    with (ROOT/'private/events.jsonl').open('a') as f:f.write(json.dumps({'time':time.time(),**CONTEXT,'kind':kind,**value},default=str)+'\n')


def guard():
    if time.time()>=BUDGET['hard_deadline_epoch']:raise TimeoutError('Original experiment deadline reached')
    if (ROOT/'STOPPED.json').exists() or (ROOT/'BUDGET_EXHAUSTED.json').exists():raise RuntimeError('Independent resource/call guard stopped')


def watchdog():
    while True:
        own=int(subprocess.check_output(['du','-sb',str(ROOT)]).split()[0]);base=int(subprocess.check_output(['du','-sb',str(BASE)]).split()[0]);total=own+base
        save('RUN_GUARD.json',{'time':time.time(),'own_bytes':own,'aggregate_bytes':total,'deadline':BUDGET['hard_deadline_epoch']})
        if own>=BUDGET['own_stop_new_bytes'] or total>=BUDGET['aggregate_stop_new_bytes'] or time.time()>=BUDGET['hard_deadline_epoch']:
            save('STOPPED.json',{'time':time.time(),'own_bytes':own,'aggregate_bytes':total,'context':CONTEXT});os.killpg(os.getpgrp(),signal.SIGTERM);return
        time.sleep(15)


rt.record=record;rt.guard=guard;rt.CONTEXT=CONTEXT


def memory_context(memory,question,train_ids):
    from hcl.memory import _memory_candidate_text
    snap=memory.snapshot()
    assert not snap['anchor'] and not snap['skills']
    assert all(r['internal_source']['task_id'] in train_ids for r in snap['raw'])
    pool=memory.raw.retrieval_candidates()+memory.abstract.retrieval_candidates()
    tok=lambda x:set(re.findall(r'[a-z0-9]+',x.lower()));query=tok(question)
    def rank(item):
        terms=tok(_memory_candidate_text(item));u=query|terms
        return (-(len(query&terms)/len(u) if u else 0),str(item['memory_id']))
    chosen=sorted(pool,key=rank)[:2]
    if not chosen:return '',[]
    return '\nHistorical training experience (different cases, not findings or answers for the current case). Use only applicable guidance; verify against current images and question:\n'+json.dumps(chosen,ensure_ascii=False),[x['memory_id'] for x in chosen]


def episode(item,state,label,tools,tool_state,Agent,prompt,memory,train_ids):
    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.memory import MemorySaver
    guard();CONTEXT.update(phase='case',episode=len(EPISODES),stage=item['stage'],state=label)
    memory.restore(state);before=memory.snapshot();context,retrieved=memory_context(memory,item['question'],train_ids)
    paths=item['images'];tool_state.update(calls=0,success=0,images=set(paths))
    model=rt.SubscriptionModel();model.deadline=min(time.time()+PLAN['limits']['per_case_seconds'],BUDGET['hard_deadline_epoch'])
    outdir=ROOT/'outputs'/str(len(EPISODES));outdir.mkdir(exist_ok=False)
    agent=Agent(model,tools,system_prompt=prompt,checkpointer=MemorySaver(),log_dir=str(outdir))
    config={'configurable':{'thread_id':'1'},'recursion_limit':2*PLAN['limits']['per_case_controller_calls']+2}
    figure=''.join(Path(p).name+' located at '+p+'\n' for p in paths)
    text='Answer this question correctly using chain of thought reasoning and carefully evaluating choices. Solve using our own vision and reasoning and thenuse tools to complement your reasoning. Trust your own judgement over any tools.\n'+item['question']+'\n'+figure+context
    content=[{'type':'text','text':text}]+[{'type':'image_url','image_url':{'url':p}} for p in paths]
    result={'id':item['id'],'stage':item['stage'],'role':item['role'],'state':label,'status':'failed','predicted':None,'retrieved_ids':retrieved};started=time.time()
    try:
        out=agent.workflow.invoke({'messages':[HumanMessage(content=content)]},config)
        result['initial_final']=out['messages'][-1].content
        out=agent.workflow.invoke({'messages':[HumanMessage(content='If you had to choose the best option, only respond with the letter of choice (only one of A, B, C, D, E, F)')]},config)
        answer=out['messages'][-1].content.strip();result.update(status='completed',final=answer,predicted=answer if answer in list('ABCDEF') else None)
    except Exception as exc:result['error']=repr(exc);record('episode_error',{'error':repr(exc)})
    assert memory.snapshot()==before
    result.update(seconds=time.time()-started,controller_calls=model.calls,tool_calls=tool_state['calls'],successful_tools=tool_state['success'],context_chars=len(context))
    EPISODES.append(result);save('private/episodes.json',EPISODES);save('PROGRESS.json',{'time':time.time(),'episodes':len(EPISODES),'context':CONTEXT,'last_status':result['status']})
    guard();return result


class AbstractController:
    def generate(self,prompt,*,state):
        from langchain_core.messages import SystemMessage,HumanMessage
        guard();model=rt.SubscriptionModel();model.deadline=min(time.time()+1200,BUDGET['hard_deadline_epoch'])
        CONTEXT.update(phase='abstract_proposal',episode=None)
        out=model.invoke([SystemMessage(content='Consolidate de-identified historical training experience. Follow the supplied JSON schema, use no tools, and return only the requested JSON object.'),HumanMessage(content=prompt)])
        if out.tool_calls:raise RuntimeError('Unexpected abstraction tool call')
        record('abstract_generated',{'text':out.content});return out.content


def proposal(memory,before,examples,predictions,stage):
    memory.restore(before)
    for interaction_index,(example,prediction) in enumerate(zip(examples,predictions)):
        memory.record_training_interaction('medical_agent',example,{'task_id':example['task_id'],'answer':prediction['predicted']})
        save('private/training_memory_'+stage+'_'+str(interaction_index)+'.json',memory.snapshot())
    memory.consolidate_task('medical_agent',examples)
    proposed=memory.snapshot();save('private/proposal_'+stage+'.json',proposed)
    candidate=local_bank(before,proposed,examples,0 if stage=='A' else 1)
    save('private/candidate_'+stage+'.json',candidate);memory.restore(before)
    record('local_candidate',{'stage':stage,'raw_before':len(before['raw']),'raw_after':len(candidate['raw']),'new_raw':sum(r['memory_id'] not in {x['memory_id'] for x in before['raw']} for r in candidate['raw']),'abstract_after':len(candidate['abstract'])})
    return candidate


def final_report():
    # Called only once candidate construction and prediction have ended, including failure.
    gold=json.loads((ROOT/'private/sealed_report_gold.json').read_text())
    report_items=[r for r in INPUTS if r['role']=='report'];keys={s:[r['id'] for r in report_items if r['stage']==s] for s in ['A','B']}
    names=['H0_static','H_A','H_AB_local_zero','H_AB_local_infinite'];results={};summary={}
    for name in names:
        physical=STATE_IDS.get(name,name);by={r['id']:r for r in EPISODES if r['role']=='report' and r['state']==physical}
        rows=[dict(by.get(r['id'],{'id':r['id'],'stage':r['stage'],'role':'report','status':'not_run','predicted':None,'reason':FATAL or 'state_unavailable'}),reported_state=name) for r in report_items]
        for r in rows:r['correct']=r.get('predicted')==gold[r['id']]
        results[name]=rows;summary[name]={s:{'correct':sum(r['correct'] for r in rows if r['stage']==s),'denominator':len(keys[s]),'completed':sum(r['status']=='completed' for r in rows if r['stage']==s),'valid_final':sum(r.get('predicted') in list('ABCDEF') for r in rows if r['stage']==s)} for s in ['A','B']}
    pairs={}
    for before,after in [('H0_static','H_A'),('H_A','H_AB_local_zero'),('H_A','H_AB_local_infinite'),('H0_static','H_AB_local_zero'),('H_AB_local_zero','H_AB_local_infinite')]:
        pairs[before+' -> '+after]={s:paired(results[before],results[after],keys[s],gold) for s in ['A','B']}
    correct=lambda n,s:{r['id'] for r in results[n] if r['stage']==s and r['correct']}
    acquired=correct('H_A','A')-correct('H0_static','A');base=correct('H0_static','A');retention={}
    for name in names[2:]:retention[name]={'acquired_A_denominator':len(acquired),'acquired_A_lost':len(acquired-correct(name,'A')),'base_A_denominator':len(base),'base_A_lost':len(base-correct(name,'A'))}
    pA=pairs['H0_static -> H_A']['A'];pB=pairs['H_A -> H_AB_local_zero']['B'];pFinal=pairs['H0_static -> H_AB_local_zero'];ret=retention['H_AB_local_zero']
    complete=all(x['completed']==x['denominator'] and x['valid_final']==x['denominator'] for stage in summary.values() for x in stage.values())
    joint=not FATAL and complete and pA['gain']>pA['loss'] and pB['gain']>pB['loss'] and len(acquired)>0 and ret['acquired_A_lost']==0 and ret['base_A_lost']==0 and sum(x['gain']-x['loss'] for x in pFinal.values())>=2
    value={'time':time.time(),'fatal':FATAL,'full_report_cases_per_state':48,'physical_case_episodes':len(EPISODES),'state_aliases':STATE_IDS,'metrics':summary,'paired':pairs,'retention':retention,'decisions':DECISIONS,'joint_development_signal':bool(joint),'clinical_or_confirmatory_efficacy':False,'memory_occupancy':{n:{k:len(v) for k,v in s.items()} for n,s in STATES.items()},'seconds_since_original_start':time.time()-BUDGET['started_epoch']}
    b=DECISIONS.get('B',{});value['gate_contribution_pattern']=bool(b.get('local_infinite_accept') and not b.get('local_zero_accept') and b.get('anchor_losses',0)>0 and pairs['H_A -> H_AB_local_infinite']['A']['loss']>0)
    save('private/report_results.json',results);save('AGGREGATE_REPORT.json',value);save('FINISHED.json',value)


def main():
    global INPUTS,FATAL
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(TimeoutError('Independent watchdog termination')))
    threading.Thread(target=watchdog,daemon=True).start()
    save('STARTED.json',{'time':time.time(),'original_start':BUDGET['started_epoch'],'deadline':BUDGET['hard_deadline_epoch']})
    INPUTS=json.loads((ROOT/'private/inputs.json').read_text());supervision=json.loads((ROOT/'private/supervision.json').read_text())
    train_ids={r['id'] for r in INPUTS if r['role']=='train'}
    from hcl.memory import ExperienceMemory,MemoryConfig
    memory=ExperienceMemory(MemoryConfig(record_dir=str(ROOT/'private/memory'),raw_capacity_per_task=10,abstract_capacity_per_task=2,retrieval_top_k=2,abstract_mode='llm',abstract_max_new_tokens=512,skill_enabled=False,anchor_selection_mode='off'),model=AbstractController())
    try:
        free=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
        available={int(x.split(',')[0]):int(x.split(',')[1]) for x in free.strip().splitlines()};gpu=int(os.environ['CUDA_VISIBLE_DEVICES']);save('GPU_PREFLIGHT.json',{'time':time.time(),'gpu':gpu,'free_mib':available[gpu]})
        if available[gpu]<45000:raise RuntimeError('Insufficient GPU headroom; no unrelated process touched')
        CONTEXT['phase']='loading_six_tools';tools,state,Agent,prompt=rt.load_six();save('TOOLS_LOADED.json',{'time':time.time(),'names':[t.name for t in tools]})
        def run(rows,snapshot,label):return [episode(r,snapshot,label,tools,state,Agent,prompt,memory,train_ids) for r in rows]
        def rows(stage,role):return [r for r in INPUTS if r['stage']==stage and r['role']==role]
        for stage in ['A','B']:
            before=STATES['H0_static' if stage=='A' else 'H_A'];sel=rows(stage,'selection');anchors=rows('A','anchor') if stage=='B' else []
            reference=run(sel,before,stage+'_reference_selection');anchor_reference=run(anchors,before,stage+'_reference_anchor') if anchors else None
            train=rows(stage,'train');pred=run(train,before,stage+'_train')
            if any(r['status']!='completed' or r['predicted'] is None for r in pred):raise RuntimeError('Training episode incomplete; no proposal fabricated')
            examples=[{'task_id':r['id'],'task_name':'medical_agent','task_type':'medical_agent','question':r['question'],'answer':supervision[r['id']],'split':'train','image_path':r['images'][0]} for r in train]
            candidate=proposal(memory,before,examples,pred,stage)
            after=run(sel,candidate,stage+'_candidate_selection');anchor_after=run(anchors,candidate,stage+'_candidate_anchor') if anchors else None
            decision=gate(reference,after,supervision,anchor_reference,anchor_after);DECISIONS[stage]=decision
            if stage=='A':STATES['H_A']=copy.deepcopy(candidate if decision['local_zero_accept'] else before)
            else:
                STATES['H_AB_local_zero']=copy.deepcopy(candidate if decision['local_zero_accept'] else before)
                STATES['H_AB_local_infinite']=copy.deepcopy(candidate if decision['local_infinite_accept'] else before)
            save('private/states.json',STATES);save('DECISIONS.json',DECISIONS)
        unique={}
        for name,snapshot in STATES.items():
            match=next((k for k,v in unique.items() if v==snapshot),None)
            STATE_IDS[name]=match or name
            if match is None:unique[name]=snapshot
        save('STATES_FROZEN.json',{'time':time.time(),'aliases':STATE_IDS,'report_gold_not_read':True,'states':list(STATES)})
        report_items=[r for r in INPUTS if r['role']=='report'];names=list(unique)
        for i,item in enumerate(report_items):
            offset=i%len(names)
            for name in names[offset:]+names[:offset]:episode(item,unique[name],name,tools,state,Agent,prompt,memory,train_ids)
    except BaseException as exc:
        FATAL=repr(exc);save('FAILED.json',{'time':time.time(),'error':FATAL,'traceback':traceback.format_exc(),'context':CONTEXT})
    finally:final_report()
    print(json.dumps({'finished':True,'fatal':FATAL,'episodes':len(EPISODES)}),flush=True)

if __name__=='__main__':main()
