"""Two image-conditioned training notes, then one frozen three-arm report matrix."""
import copy
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
import traceback
import run as core
from interface import select,render_views,option_texts
from notes_logic import ARMS,FIELDS,parse_note
from logic import paired
from storage import tree_bytes

ROOT=core.ROOT;BUDGET=core.BUDGET
INPUTS=[];SOURCE={};STATES={};FATAL=None;NOTE_COUNT=0
os.environ['TMPDIR']='/data/bmw/workers/t1002';os.environ['TRITON_CACHE_DIR']='/data/bmw/workers/t1002/triton'


def watchdog():
    roots=[ROOT]+[ROOT.parent/x for x in ['medrax_subscription_v4','medrax_hcl_v5','medrax_memory_interface_v6','medrax_corrective_memory_v7']]
    while True:
        try:
            sizes=[tree_bytes(p) for p in roots];total=sum(sizes)
            core.save('RUN_GUARD.json',{'time':time.time(),'own_bytes':sizes[0],'aggregate_bytes':total,'deadline':BUDGET['hard_deadline_epoch'],'probe':'python_lstat_no_inner_symlinks'})
            if sizes[0]>=BUDGET['own_stop_new_bytes'] or total>=BUDGET['aggregate_stop_new_bytes'] or time.time()>=BUDGET['hard_deadline_epoch']:raise RuntimeError('Storage or original wall-clock limit')
        except BaseException as exc:
            try:core.save('STOPPED.json',{'time':time.time(),'error':repr(exc),'context':core.CONTEXT})
            finally:os.killpg(os.getpgrp(),signal.SIGTERM)
            return
        time.sleep(15)


def context(memory,question,train_ids):
    snap=memory.snapshot();assert not snap['anchor'] and not snap['skills'] and not (snap['raw'] and snap['abstract'])
    assert all(r['internal_source']['task_id'] in train_ids for r in snap['raw']+snap['abstract'])
    if snap['raw']:
        chosen=select(memory.raw.retrieval_candidates(),[],question,'global_semantic');views=render_views(chosen,snap['raw'],'global_semantic');ids=[x['memory_id'] for x in chosen]
    else:
        chosen=snap['abstract'];views=[{'memory_id':x['memory_id'],'memory_kind':'abstract',**x['note']} for x in chosen];ids=[x['memory_id'] for x in chosen]
    if not views:return '',[]
    return '\nHistorical training experience (different cases, not findings or answers for the current case). Use only applicable guidance; verify against current images and question:\n'+json.dumps(views,ensure_ascii=False),ids


core.memory_context=context


def generate_notes():
    global NOTE_COUNT
    from langchain_core.messages import HumanMessage,SystemMessage
    payload={x['source_id']:x for x in json.loads((ROOT/'private/source_training.json').read_text())};notes=[]
    for i,record in enumerate(SOURCE['raw']):
        source=payload[record['internal_source']['task_id']];mapping=option_texts(record['question']);core.CONTEXT.update(phase='error_abstraction',episode=None,stage='B',state='note_generation')
        instruction='Learn one conditional correction from this supervised historical training error. Inspect the attached original training images and question. The supplied target is training supervision; do not invent visual findings to justify it. Distinguish observable evidence from uncertain interpretation. State a reusable correction and when it must not be applied. Do not include option letters, answer-output-format directives, identifiers or personal information. Return only a JSON object with exactly these six nonempty string fields: '+', '.join(FIELDS)+'. Total field text must be at most 1800 characters. No tools. These are candidate notes, not clinically validated knowledge.'
        data={'training_question':source['question'],'initial_training_response':source['initial_response'],'mistaken_option_text':mapping[record['response']],'supervised_target_text':mapping[record['feedback']]}
        content=[{'type':'text','text':json.dumps(data,ensure_ascii=False)}]+[{'type':'image_url','image_url':{'url':p}} for p in source['images']]
        model=core.rt.SubscriptionModel();model.deadline=min(time.time()+1200,BUDGET['hard_deadline_epoch']);out=model.invoke([SystemMessage(content=instruction),HumanMessage(content=content)])
        core.save('private/note_generation_'+str(i)+'.json',{'time':time.time(),'source_id':source['source_id'],'raw_output':out.content,'tool_calls':out.tool_calls,'images':len(source['images'])})
        if out.tool_calls:raise ValueError('Unexpected tool call during note construction')
        note=parse_note(out.content);notes.append({'memory_id':'abstract_error_'+str(i),'memory_type':'abstract','abstract_role':'failure_avoidance','internal_source':{'task_id':source['source_id'],'task_name':'medical_agent','split':'train'},'note':note});NOTE_COUNT=len(notes);core.save('private/notes.json',notes)
    return notes


def finalize():
    gold=json.loads((ROOT/'private/sealed_report_gold.json').read_text());keys={s:[x['id'] for x in INPUTS if x['stage']==s] for s in ['A','B']};all_keys=[x['id'] for x in INPUTS];results={};metrics={}
    for arm in ARMS:
        by={x['id']:x for x in core.EPISODES if x['state']==arm};rows=[dict(by.get(x['id'],{'id':x['id'],'stage':x['stage'],'status':'not_run','predicted':None,'reason':FATAL or 'missing_episode'})) for x in INPUTS]
        for x in rows:x['correct']=x.get('predicted')==gold[x['id']]
        results[arm]=rows;metrics[arm]={s:{'correct':sum(x['correct'] for x in rows if x['stage']==s),'denominator':len(keys[s]),'completed':sum(x['status']=='completed' for x in rows if x['stage']==s),'valid_final':sum(x.get('predicted') in list('ABCDEF') for x in rows if x['stage']==s)} for s in ['A','B']}
        parsed=[]
        for x in rows:
            m=re.match(r'([A-F])(?:[).:\s]|$)',x.get('initial_final','').lstrip().lstrip('*'))
            if m:parsed.append(m.group(1)!=x.get('predicted'))
        metrics[arm]['initial_final_consistency']={'parseable':len(parsed),'changed_option':sum(parsed)}
        actual=[x for x in core.EPISODES if x['state']==arm];ids=[i for x in actual for i in x['retrieved_ids']]
        metrics[arm]['context']={'mean_chars':sum(x['context_chars'] for x in actual)/len(actual) if actual else None,'memory_reads':len(ids),'distinct_records':len(set(ids))}
    pairs={a+' -> '+b:{s:paired(results[a],results[b],kk,gold) for s,kk in [*keys.items(),('pooled',all_keys)]} for a,b in [('H0_static','raw_errors'),('H0_static','error_notes'),('raw_errors','error_notes')]}
    complete=len(core.EPISODES)==144 and all(x['status']=='completed' and x.get('predicted') in list('ABCDEF') for x in core.EPISODES)
    base=pairs['H0_static -> error_notes'];raw=pairs['raw_errors -> error_notes'];signal_ok=not FATAL and NOTE_COUNT==2 and complete and base['B']['gain']-base['B']['loss']>=2 and raw['B']['gain']-raw['B']['loss']>=2 and base['pooled']['loss']==0
    value={'time':time.time(),'fatal':FATAL,'planned_report_episodes':144,'physical_report_episodes':len(core.EPISODES),'unique_report_cases':48,'valid_notes':NOTE_COUNT,'planned_note_calls':2,'metrics':metrics,'paired':pairs,'development_signal':bool(signal_ok),'notes_medically_validated':False,'continual_learning_or_retention_established':False,'seconds_since_original_start':time.time()-BUDGET['started_epoch']}
    core.save('private/report_results.json',results);core.save('AGGREGATE_REPORT.json',value);core.save('FINISHED.json',value)


def main():
    global INPUTS,SOURCE,STATES,FATAL
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(TimeoutError('Independent watchdog termination')));threading.Thread(target=watchdog,daemon=True).start()
    core.save('STARTED.json',{'time':time.time(),'original_start':BUDGET['started_epoch'],'deadline':BUDGET['hard_deadline_epoch']})
    INPUTS=json.loads((ROOT/'private/inputs.json').read_text());SOURCE=json.loads((ROOT/'private/source_state.json').read_text());train_ids={r['internal_source']['task_id'] for r in SOURCE['raw']}
    from hcl.memory import ExperienceMemory,MemoryConfig
    memory=ExperienceMemory(MemoryConfig(record_dir=str(ROOT/'private/memory'),raw_capacity_per_task=4,abstract_capacity_per_task=2,skill_enabled=False,anchor_selection_mode='off'))
    try:
        notes=generate_notes();STATES={'H0_static':copy.deepcopy(core.EMPTY),'raw_errors':copy.deepcopy(SOURCE),'error_notes':{**copy.deepcopy(core.EMPTY),'abstract':notes}}
        for state in STATES.values():
            memory.restore(state)
            for item in INPUTS:context(memory,item['question'],train_ids)
        core.save('private/states.json',STATES);core.save('STATES_FROZEN.json',{'time':time.time(),'report_gold_not_read':True,'valid_notes':2,'source_B_training_errors':2,'arms':ARMS})
        free=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True);available={int(x.split(',')[0]):int(x.split(',')[1]) for x in free.strip().splitlines()};gpu=int(os.environ['CUDA_VISIBLE_DEVICES']);core.save('GPU_PREFLIGHT.json',{'time':time.time(),'gpu':gpu,'free_mib':available[gpu]})
        if available[gpu]<45000:raise RuntimeError('Insufficient GPU headroom; no unrelated process touched')
        core.CONTEXT['phase']='loading_six_tools';tools,state,Agent,prompt=core.rt.load_six();core.save('TOOLS_LOADED.json',{'time':time.time(),'names':[t.name for t in tools]})
        for i,item in enumerate(INPUTS):
            offset=i%len(ARMS)
            for arm in ARMS[offset:]+ARMS[:offset]:core.episode(item,STATES[arm],arm,tools,state,Agent,prompt,memory,train_ids)
        assert STATES==json.loads((ROOT/'private/states.json').read_text()) and SOURCE==json.loads((ROOT/'private/source_state.json').read_text())
    except BaseException as exc:FATAL=repr(exc);core.save('FAILED.json',{'time':time.time(),'error':FATAL,'traceback':traceback.format_exc(),'context':core.CONTEXT})
    finally:finalize()
    print(json.dumps({'finished':True,'fatal':FATAL,'episodes':len(core.EPISODES)}),flush=True)

if __name__=='__main__':main()
