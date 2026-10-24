"""Reuse v5 native episodes; change only the frozen memory interface factors."""
import collections
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
from interface import ARMS,select,render_views
from logic import paired

ROOT=core.ROOT;BUDGET=core.BUDGET;BASE=core.BASE;PREVIOUS=Path(os.environ['R998_PREVIOUS'])
CURRENT='H0_static';INPUTS=[];SOURCE={};FATAL=None;EXPOSURES=[]
os.environ['TMPDIR']='/data/bmw/workers/t998';os.environ['TRITON_CACHE_DIR']='/data/bmw/workers/t998/triton'


def watchdog():
    while True:
        sizes={name:int(subprocess.check_output(['du','-sb',str(path)]).split()[0]) for name,path in [('own',ROOT),('base',BASE),('previous',PREVIOUS)]};total=sum(sizes.values())
        core.save('RUN_GUARD.json',{'time':time.time(),'own_bytes':sizes['own'],'aggregate_bytes':total,'deadline':BUDGET['hard_deadline_epoch']})
        if sizes['own']>=BUDGET['own_stop_new_bytes'] or total>=BUDGET['aggregate_stop_new_bytes'] or time.time()>=BUDGET['hard_deadline_epoch']:
            core.save('STOPPED.json',{'time':time.time(),'bytes':sizes,'context':core.CONTEXT});os.killpg(os.getpgrp(),signal.SIGTERM);return
        time.sleep(15)


def context(memory,question,train_ids):
    snap=memory.snapshot();assert all(r['internal_source']['task_id'] in train_ids for r in snap['raw']) and not snap['anchor'] and not snap['skills']
    chosen=select(memory.raw.retrieval_candidates(),memory.abstract.retrieval_candidates(),question,CURRENT);views=render_views(chosen,snap['raw'],CURRENT)
    ids=[r['memory_id'] for r in chosen];EXPOSURES.append({'episode':core.CONTEXT['episode'],'arm':CURRENT,'ids':ids,'kinds':[r['memory_kind'] for r in chosen]})
    if not views:return '',[]
    return '\nHistorical training experience (different cases, not findings or answers for the current case). Use only applicable guidance; verify against current images and question:\n'+json.dumps(views,ensure_ascii=False),ids


core.memory_context=context


def finalize():
    gold=json.loads((ROOT/'private/sealed_report_gold.json').read_text());results={};metrics={};keys={s:[x['id'] for x in INPUTS if x['stage']==s] for s in ['A','B']};all_keys=[x['id'] for x in INPUTS]
    for arm in ARMS:
        by={x['id']:x for x in core.EPISODES if x['state']==arm};rows=[dict(by.get(x['id'],{'id':x['id'],'stage':x['stage'],'status':'not_run','predicted':None,'reason':FATAL or 'missing_episode'})) for x in INPUTS]
        for x in rows:x['correct']=x.get('predicted')==gold[x['id']]
        results[arm]=rows;metrics[arm]={s:{'correct':sum(x['correct'] for x in rows if x['stage']==s),'denominator':12,'completed':sum(x['status']=='completed' for x in rows if x['stage']==s),'valid_final':sum(x.get('predicted') in list('ABCDEF') for x in rows if x['stage']==s)} for s in ['A','B']}
        parsed=flips=0
        for x in rows:
            m=re.match(r'([A-F])[).:\s]',x.get('initial_final','').lstrip().lstrip('*'))
            if m:parsed+=1;flips+=m.group(1)!=x.get('predicted')
        metrics[arm]['initial_final_consistency']={'parseable':parsed,'changed_option':flips}
        actual=[x for x in core.EPISODES if x['state']==arm];ex=[e for e in EXPOSURES if e['arm']==arm]
        metrics[arm]['context']={'mean_chars':sum(x['context_chars'] for x in actual)/len(actual) if actual else None,'memory_kinds':dict(collections.Counter(k for e in ex for k in e['kinds']))}
    contrasts=[('global_letter','balanced_letter'),('global_semantic','balanced_semantic'),('global_letter','global_semantic'),('balanced_letter','balanced_semantic'),('global_letter','balanced_semantic')]+[('H0_static',arm) for arm in ARMS[1:]]
    pairs={a+' -> '+b:{s:paired(results[a],results[b],kk,gold) for s,kk in [*keys.items(),('pooled',all_keys)]} for a,b in contrasts}
    complete=all(v[s]['completed']==v[s]['valid_final']==12 for v in metrics.values() for s in ['A','B']);combined=pairs['global_letter -> balanced_semantic']['pooled'];vsbase=pairs['H0_static -> balanced_semantic']
    signal_ok=not FATAL and complete and combined['gain']-combined['loss']>=2 and combined['loss']==0 and all(vsbase[s]['gain']>=vsbase[s]['loss'] for s in ['A','B'])
    agreement={}
    for retrieval in ['global','balanced']:
        a={x['id']:x['retrieved_ids'] for x in core.EPISODES if x['state']==retrieval+'_letter'};b={x['id']:x['retrieved_ids'] for x in core.EPISODES if x['state']==retrieval+'_semantic'}
        agreement[retrieval]={'paired_cases':len(a.keys()&b.keys()),'same_ids_order':all(a[k]==b[k] for k in a.keys()&b.keys())}
    value={'time':time.time(),'fatal':FATAL,'planned_episodes':120,'physical_episodes':len(core.EPISODES),'unique_report_cases':24,'metrics':metrics,'paired':pairs,'representation_arm_retrieval_identity':agreement,'combined_development_signal':bool(signal_ok),'continual_learning_or_retention_established':False,'seconds_since_original_start':time.time()-BUDGET['started_epoch']}
    core.save('private/report_results.json',results);core.save('private/exposures.json',EXPOSURES);core.save('AGGREGATE_REPORT.json',value);core.save('FINISHED.json',value)


def main():
    global CURRENT,INPUTS,SOURCE,FATAL
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(TimeoutError('Independent watchdog termination')));threading.Thread(target=watchdog,daemon=True).start()
    core.save('STARTED.json',{'time':time.time(),'original_start':BUDGET['started_epoch'],'deadline':BUDGET['hard_deadline_epoch']})
    INPUTS=json.loads((ROOT/'private/inputs.json').read_text());SOURCE=json.loads((ROOT/'private/source_state.json').read_text());train_ids={r['internal_source']['task_id'] for r in SOURCE['raw']}
    from hcl.memory import ExperienceMemory,MemoryConfig
    memory=ExperienceMemory(MemoryConfig(record_dir=str(ROOT/'private/memory'),raw_capacity_per_task=10,abstract_capacity_per_task=2,skill_enabled=False,anchor_selection_mode='off'))
    try:
        free=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True);available={int(x.split(',')[0]):int(x.split(',')[1]) for x in free.strip().splitlines()};gpu=int(os.environ['CUDA_VISIBLE_DEVICES']);core.save('GPU_PREFLIGHT.json',{'time':time.time(),'gpu':gpu,'free_mib':available[gpu]})
        if available[gpu]<45000:raise RuntimeError('Insufficient GPU headroom; no unrelated process touched')
        core.CONTEXT['phase']='loading_six_tools';tools,state,Agent,prompt=core.rt.load_six();core.save('TOOLS_LOADED.json',{'time':time.time(),'names':[t.name for t in tools]})
        for i,item in enumerate(INPUTS):
            offset=i%len(ARMS)
            for arm in ARMS[offset:]+ARMS[:offset]:
                CURRENT=arm;snapshot=core.EMPTY if arm=='H0_static' else SOURCE
                core.episode(item,snapshot,arm,tools,state,Agent,prompt,memory,train_ids)
        assert SOURCE==json.loads((ROOT/'private/source_state.json').read_text())
    except BaseException as exc:FATAL=repr(exc);core.save('FAILED.json',{'time':time.time(),'error':FATAL,'traceback':traceback.format_exc(),'context':core.CONTEXT})
    finally:finalize()
    print(json.dumps({'finished':True,'fatal':FATAL,'episodes':len(core.EPISODES)}),flush=True)

if __name__=='__main__':main()
