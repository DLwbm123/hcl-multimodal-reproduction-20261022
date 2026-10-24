"""Train once without memory, freeze matched banks, then evaluate all report cells."""
import collections
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
from banks import ARMS,EMPTY,matched_banks
from interface import select,render_views
from logic import paired
from storage import tree_bytes

ROOT=core.ROOT;BUDGET=core.BUDGET
INPUTS=[];STATES={};BANK_AUDIT={};FATAL=None
os.environ['TMPDIR']='/data/bmw/workers/t1000';os.environ['TRITON_CACHE_DIR']='/data/bmw/workers/t1000/triton'


def watchdog():
    roots=[ROOT]+[ROOT.parent/x for x in ['medrax_subscription_v4','medrax_hcl_v5','medrax_memory_interface_v6']]
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
    snap=memory.snapshot();assert not snap['abstract'] and not snap['anchor'] and not snap['skills']
    assert all(r['internal_source']['task_id'] in train_ids for r in snap['raw'])
    chosen=select(memory.raw.retrieval_candidates(),[],question,'global_semantic');views=render_views(chosen,snap['raw'],'global_semantic')
    if not views:return '',[]
    return '\nHistorical training experience (different cases, not findings or answers for the current case). Use only applicable guidance; verify against current images and question:\n'+json.dumps(views,ensure_ascii=False),[x['memory_id'] for x in chosen]


core.memory_context=context


def finalize():
    gold=json.loads((ROOT/'private/sealed_report_gold.json').read_text());items=[x for x in INPUTS if x['role']=='report'];keys={s:[x['id'] for x in items if x['stage']==s] for s in ['A','B']};all_keys=[x['id'] for x in items];results={};metrics={}
    for arm in ARMS:
        by={x['id']:x for x in core.EPISODES if x['state']==arm and x['role']=='report'}
        rows=[dict(by.get(x['id'],{'id':x['id'],'stage':x['stage'],'status':'not_run','predicted':None,'reason':FATAL or 'missing_episode'})) for x in items]
        for x in rows:x['correct']=x.get('predicted')==gold[x['id']]
        results[arm]=rows;metrics[arm]={s:{'correct':sum(x['correct'] for x in rows if x['stage']==s),'denominator':len(keys[s]),'completed':sum(x['status']=='completed' for x in rows if x['stage']==s),'valid_final':sum(x.get('predicted') in list('ABCDEF') for x in rows if x['stage']==s)} for s in ['A','B']}
        parsed=[]
        for x in rows:
            m=re.match(r'([A-F])(?:[).:\s]|$)',x.get('initial_final','').lstrip().lstrip('*'))
            if m:parsed.append(m.group(1)!=x.get('predicted'))
        metrics[arm]['initial_final_consistency']={'parseable':len(parsed),'changed_option':sum(parsed)}
        actual=[x for x in core.EPISODES if x['state']==arm];ids=[i for x in actual for i in x['retrieved_ids']]
        metrics[arm]['context']={'mean_chars':sum(x['context_chars'] for x in actual)/len(actual) if actual else None,'raw_reads':len(ids),'distinct_records':len(set(ids))}
    contrasts=[('H0_static','success_memory'),('H0_static','corrective_memory'),('success_memory','corrective_memory')]
    pairs={a+' -> '+b:{s:paired(results[a],results[b],kk,gold) for s,kk in [*keys.items(),('pooled',all_keys)]} for a,b in contrasts}
    complete=len(core.EPISODES)==176 and all(x['status']=='completed' and x.get('predicted') in list('ABCDEF') for x in core.EPISODES)
    coverage=all(BANK_AUDIT.get(s,{}).get('matched_selected_per_bank',0)>=1 for s in ['A','B']);base=pairs['H0_static -> corrective_memory'];success=pairs['success_memory -> corrective_memory']
    signal_ok=not FATAL and complete and coverage and base['pooled']['gain']-base['pooled']['loss']>=2 and base['pooled']['loss']==0 and success['pooled']['gain']-success['pooled']['loss']>=2 and all(success[s]['gain']>=success[s]['loss'] for s in ['A','B'])
    train=[x for x in core.EPISODES if x['role']=='train']
    value={'time':time.time(),'fatal':FATAL,'planned_episodes':176,'physical_episodes':len(core.EPISODES),'training_completed':len(train),'planned_training_episodes':32,'unique_report_cases':48,'metrics':metrics,'paired':pairs,'training_bank_audit':BANK_AUDIT,'source_error_coverage':coverage,'acquisition_development_signal':bool(signal_ok),'continual_learning_or_retention_established':False,'seconds_since_original_start':time.time()-BUDGET['started_epoch']}
    core.save('private/report_results.json',results);core.save('AGGREGATE_REPORT.json',value);core.save('FINISHED.json',value)


def main():
    global INPUTS,STATES,BANK_AUDIT,FATAL
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(TimeoutError('Independent watchdog termination')));threading.Thread(target=watchdog,daemon=True).start()
    core.save('STARTED.json',{'time':time.time(),'original_start':BUDGET['started_epoch'],'deadline':BUDGET['hard_deadline_epoch']})
    INPUTS=json.loads((ROOT/'private/inputs.json').read_text());supervision=json.loads((ROOT/'private/supervision.json').read_text());train=[x for x in INPUTS if x['role']=='train'];train_ids={x['id'] for x in train}
    assert set(supervision)==train_ids
    from hcl.memory import ExperienceMemory,MemoryConfig
    memory=ExperienceMemory(MemoryConfig(record_dir=str(ROOT/'private/memory'),raw_capacity_per_task=32,abstract_capacity_per_task=2,skill_enabled=False,anchor_selection_mode='off'))
    try:
        free=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True);available={int(x.split(',')[0]):int(x.split(',')[1]) for x in free.strip().splitlines()};gpu=int(os.environ['CUDA_VISIBLE_DEVICES']);core.save('GPU_PREFLIGHT.json',{'time':time.time(),'gpu':gpu,'free_mib':available[gpu]})
        if available[gpu]<45000:raise RuntimeError('Insufficient GPU headroom; no unrelated process touched')
        core.CONTEXT['phase']='loading_six_tools';tools,state,Agent,prompt=core.rt.load_six();core.save('TOOLS_LOADED.json',{'time':time.time(),'names':[t.name for t in tools]})
        predictions=[core.episode(x,EMPTY,'training_H0',tools,state,Agent,prompt,memory,train_ids) for x in train]
        if any(x['status']!='completed' or x.get('predicted') not in list('ABCDEF') for x in predictions):raise RuntimeError('Training incomplete; no bank fabricated')
        memory.restore(EMPTY)
        for i,(x,pred) in enumerate(zip(train,predictions)):
            example={'task_id':x['id'],'task_name':'medical_agent','task_type':'medical_agent','question':x['question'],'answer':supervision[x['id']],'split':'train','image_path':x['images'][0]}
            memory.record_training_interaction('medical_agent',example,{'task_id':x['id'],'answer':pred['predicted']});core.save('private/training_memory_'+str(i)+'.json',memory.snapshot())
        pool=memory.snapshot();assert len(pool['raw'])==32
        core.save('private/training_pool.json',pool);STATES,BANK_AUDIT=matched_banks(pool['raw'],INPUTS,predictions,supervision)
        for arm,snapshot in STATES.items():
            memory.restore(snapshot)
            for item in INPUTS:context(memory,item['question'],train_ids)  # Validate actual option mappings before any report inference.
        core.save('private/states.json',STATES);core.save('BANK_AUDIT.json',BANK_AUDIT);core.save('STATES_FROZEN.json',{'time':time.time(),'report_gold_not_read':True,'bank_audit':BANK_AUDIT,'arms':ARMS})
        for i,item in enumerate(x for x in INPUTS if x['role']=='report'):
            offset=i%len(ARMS)
            for arm in ARMS[offset:]+ARMS[:offset]:core.episode(item,STATES[arm],arm,tools,state,Agent,prompt,memory,train_ids)
        assert STATES==json.loads((ROOT/'private/states.json').read_text())
    except BaseException as exc:FATAL=repr(exc);core.save('FAILED.json',{'time':time.time(),'error':FATAL,'traceback':traceback.format_exc(),'context':core.CONTEXT})
    finally:finalize()
    print(json.dumps({'finished':True,'fatal':FATAL,'episodes':len(core.EPISODES)}),flush=True)

if __name__=='__main__':main()
