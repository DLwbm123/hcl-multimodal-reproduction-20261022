"""One frozen A-acquisition test of evidence-bounded procedural memory."""
import copy, json, os, re, signal, subprocess, threading, time, traceback
from pathlib import Path
import run as core
from logic import paired
from storage import tree_bytes
from memory_logic import ARMS,FIELDS,GENERIC,parse_note,render,repeated_gate

ROOT=core.ROOT;BUDGET=core.BUDGET
INPUTS=[];STATES={};NOTES=[];FATAL=None;DECISION='not_started'
os.environ['TMPDIR']='/data/bmw/workers/t1006';os.environ['TRITON_CACHE_DIR']='/data/bmw/workers/t1006/triton'


def watchdog():
    roots=[ROOT]+[ROOT.parent/n for n in ['medrax_subscription_v4','medrax_hcl_v5','medrax_memory_interface_v6','medrax_corrective_memory_v7','medrax_error_abstraction_v8','medrax_learning_qualification_v9']]
    while True:
        try:
            sizes=[tree_bytes(p) for p in roots];core.save('RUN_GUARD.json',{'time':time.time(),'own_bytes':sizes[0],'aggregate_bytes':sum(sizes),'deadline':BUDGET['hard_deadline_epoch']})
            if sizes[0]>=BUDGET['own_stop_new_bytes'] or sum(sizes)>=BUDGET['aggregate_stop_new_bytes'] or time.time()>=BUDGET['hard_deadline_epoch']:raise RuntimeError('Original time or storage stop')
        except BaseException as exc:
            try:core.save('STOPPED.json',{'time':time.time(),'error':repr(exc),'context':core.CONTEXT})
            finally:os.killpg(os.getpgrp(),signal.SIGTERM)
            return
        time.sleep(15)


def context(memory,question,train_ids):return render(memory.snapshot(),question,train_ids)
core.memory_context=context


def option_text(question,label):
    matches=list(re.finditer(r'(?m)^\s*([A-F])[.)]\s+',question));mapping={}
    for i,m in enumerate(matches):
        assert m.group(1) not in mapping
        mapping[m.group(1)]=question[m.end():matches[i+1].start() if i+1<len(matches) else len(question)].strip()
    assert label in mapping and mapping[label]
    return mapping[label]


def generate(sources):
    global NOTES
    from langchain_core.messages import SystemMessage,HumanMessage
    records=[]
    for i,source in enumerate(sources):
        item=source['item'];core.guard();core.CONTEXT.update(phase='procedural_learning',episode=None,stage='A',state='note_generation')
        instruction=('Learn one reusable conditional PROCEDURE from this historical supervised training attempt, which may already be correct. Inspect the attached original images. The benchmark target is unadjudicated and may be ambiguous, unsupported by the input or wrong. Do not rationalize it, invent facts, or teach a specific diagnosis merely because it is the target. A note can be admitted only if its procedure is supported by the supplied observations/reasoning and remains useful when the target is uncertain. Distinguish observations, inferences, missing evidence and benchmark supervision. Otherwise set admit=false and preserve the reason. Do not include option letters, answer-format instructions, source/case identifiers or personal information. No tools. Return only JSON with exactly keys admit (boolean), reason (nonempty string <=600 characters), cue_terms (2-12 concise words or phrases, each <=60 characters), '+', '.join(FIELDS)+' (nonempty strings with total <=1600 characters). Cue terms must be concrete features or reasoning tasks for which the procedure applies; avoid generic chest/image/answer words. Do not claim expert validation. There is exactly one generation; no retries.')
        payload={'historical_question':item['question'],'historical_attempt':source['attempt']['initial_final'],'attempt_option_text':option_text(item['question'],source['attempt']['predicted']),'unadjudicated_benchmark_target_text':option_text(item['question'],source['gold']),'benchmark_attempt_correct':source['attempt']['predicted']==source['gold']}
        content=[{'type':'text','text':json.dumps(payload,ensure_ascii=False)}]+[{'type':'image_url','image_url':{'url':p}} for p in item['images']]
        model=core.rt.SubscriptionModel();model.deadline=min(time.time()+1200,BUDGET['hard_deadline_epoch']);out=model.invoke([SystemMessage(content=instruction),HumanMessage(content=content)])
        core.save('private/generation_'+str(i)+'.json',{'time':time.time(),'source_id':item['id'],'raw_output':out.content,'tool_calls':out.tool_calls,'images':len(item['images'])})
        if out.tool_calls:raise ValueError('Unexpected tools in note generation')
        parsed=parse_note(out.content);NOTES.append({'source_id':item['id'],**parsed});core.save('private/proposals.json',NOTES)
        if parsed['admit']:records.append({'memory_id':'procedure_A_'+str(i).zfill(2),'memory_type':'abstract','abstract_role':'failure_avoidance','internal_source':{'task_id':item['id'],'task_name':'medical_agent','split':'train'},'cue_terms':parsed['cue_terms'],'note':{k:parsed[k] for k in FIELDS}})
    if not records:return []
    core.CONTEXT.update(phase='procedural_consolidation',episode=None,stage='A',state='consolidation')
    instruction=('Consolidate the admitted historical procedures into exactly two complementary HCL policies: main_policy and failure_avoidance. Preserve conditional applicability, exclusions and supervision uncertainty. Do not introduce new medical facts, diagnoses, answer letters or source-specific conclusions. Return only a JSON object with exactly those two keys. Each value must have exactly source_indices (nonempty list of admitted integer indices), cue_terms (2-12 concrete words/phrases, each <=60 characters), and '+', '.join(FIELDS)+' (nonempty strings, total <=1600 characters per policy). Only cite supplied admitted indices. No tools. This is one generation without retries, not expert validation.')
    admitted=[{'source_index':i,**p} for i,p in enumerate(NOTES) if p['admit']]
    payload=[{k:v for k,v in p.items() if k!='source_id'} for p in admitted]
    model=core.rt.SubscriptionModel();model.deadline=min(time.time()+1200,BUDGET['hard_deadline_epoch'])
    out=model.invoke([SystemMessage(content=instruction),HumanMessage(content=json.dumps(payload,ensure_ascii=False))])
    core.save('private/consolidation.json',{'time':time.time(),'raw_output':out.content,'tool_calls':out.tool_calls,'admitted_source_count':len(admitted)})
    if out.tool_calls:raise ValueError('Unexpected tools in consolidation')
    text=out.content.strip()
    if text.startswith('```'):text=re.sub(r'^```(?:json)?\s*|\s*```$','',text)
    policies=json.loads(text);assert set(policies)=={'main_policy','failure_avoidance'}
    allowed={p['source_index']:p['source_id'] for p in admitted};result=[]
    for role in ['main_policy','failure_avoidance']:
        value=policies[role];assert set(value)==set(FIELDS+['source_indices','cue_terms'])
        indices=value['source_indices'];assert isinstance(indices,list) and indices and all(type(i) is int and i in allowed for i in indices) and len(indices)==len(set(indices))
        parse_note(json.dumps({'admit':True,'reason':'Consolidated from admitted procedures.','cue_terms':value['cue_terms'],**{k:value[k] for k in FIELDS}}))
        source_ids=[allowed[i] for i in indices]
        result.append({'memory_id':'procedure_A_'+role,'memory_type':'abstract','abstract_role':role,'internal_source':{'task_id':source_ids[0],'task_name':'medical_agent','split':'train'},'source_ids':source_ids,'cue_terms':value['cue_terms'],'note':{k:value[k] for k in FIELDS}})
    return result


def finalize():
    gold=json.loads((ROOT/'private/sealed_development_gold.json').read_text());ids=[r['id'] for r in INPUTS]
    all_results={};metrics={};pairs=[];refs=[];generics=[];learned=[]
    for block in range(2):
        rows_by={}
        for arm in ARMS:
            label=arm+'_r'+str(block+1);by={r['id']:r for r in core.EPISODES if r['state']==label}
            rows=[dict(by.get(r['id'],{'id':r['id'],'stage':'A','status':'not_run','predicted':None})) for r in INPUTS]
            for r in rows:r['correct']=r.get('predicted')==gold[r['id']]
            all_results[label]=rows;rows_by[arm]=rows
            actual=list(by.values());metrics[label]={'denominator':len(ids),'correct':sum(r['correct'] for r in rows),'valid':sum(r['status']=='completed' and r.get('predicted') in list('ABCDEF') for r in rows),'memory_used_episodes':sum(bool(r['retrieved_ids']) for r in actual),'mean_context_chars':sum(r['context_chars'] for r in actual)/len(actual) if actual else None,'distinct_retrieved_records':len({k for r in actual for k in r['retrieved_ids']})}
        refs.append({r['id']:r for r in rows_by['H0_static']});generics.append({r['id']:r for r in rows_by['generic_procedure']});learned.append({r['id']:r for r in rows_by['H_A']})
        pairs.append({'H0_to_HA':paired(rows_by['H0_static'],rows_by['H_A'],ids,gold),'generic_to_HA':paired(rows_by['generic_procedure'],rows_by['H_A'],ids,gold),'H0_to_generic':paired(rows_by['H0_static'],rows_by['generic_procedure'],ids,gold)})
    complete=FATAL is None and len(core.EPISODES)==144 and all(v['valid']==24 for v in metrics.values())
    gate=repeated_gate(pairs,refs,generics,learned,ids,gold,complete)
    core.save('private/acquisition_gate.json',gate);gate={k:v for k,v in gate.items() if k!='repeat_observed_acquired_ids'}
    value={'time':time.time(),'fatal':FATAL,'decision':DECISION,'planned_max_episodes':144,'physical_episodes':len(core.EPISODES),'complete':complete,'proposals_generated':len(NOTES),'proposals_admitted':sum(p['admit'] for p in NOTES),'metrics':metrics,'paired_per_repeat':pairs,'acquisition_gate':gate,'gate_headroom_per_repeat':[{'H0_reachable':24-metrics['H0_static_r'+str(i+1)]['correct']>=2,'generic_reachable':24-metrics['generic_procedure_r'+str(i+1)]['correct']>=1} for i in range(2)],'confirmation_used':False,'medical_validity_established':False,'sequential_learning_or_retention_established':False,'seconds_since_original_start':time.time()-BUDGET['started_epoch']}
    core.save('private/results.json',all_results);core.save('AGGREGATE_REPORT.json',value);core.save('FINISHED.json',value)


def main():
    global INPUTS,STATES,FATAL,DECISION
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('Independent watchdog termination')));threading.Thread(target=watchdog,daemon=True).start()
    core.save('STARTED.json',{'time':time.time(),'original_start':BUDGET['started_epoch'],'deadline':BUDGET['hard_deadline_epoch']})
    INPUTS=json.loads((ROOT/'private/inputs.json').read_text());sources=json.loads((ROOT/'private/source_training.json').read_text());train_ids={s['item']['id'] for s in sources}
    from hcl.memory import ExperienceMemory,MemoryConfig
    memory=ExperienceMemory(MemoryConfig(record_dir=str(ROOT/'private/memory'),abstract_capacity_per_task=2,skill_enabled=False,anchor_selection_mode='off'))
    try:
        records=generate(sources)
        STATES={'H0_static':copy.deepcopy(core.EMPTY),'generic_procedure':{**copy.deepcopy(core.EMPTY),'abstract':[{'memory_id':'fixed_generic','memory_type':'abstract','note':GENERIC}]},'H_A':{**copy.deepcopy(core.EMPTY),'abstract':records}}
        core.save('private/states.json',STATES);core.save('STATES_FROZEN.json',{'time':time.time(),'development_gold_not_read':True,'proposals':len(NOTES),'admitted':len(records)})
        if not records:DECISION='no_admitted_procedures';return
        for state in STATES.values():
            memory.restore(state)
            for item in INPUTS:context(memory,item['question'],train_ids)
            assert memory.snapshot()==state
        free=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True);available={int(x.split(',')[0]):int(x.split(',')[1]) for x in free.strip().splitlines()};gpu=int(os.environ['CUDA_VISIBLE_DEVICES']);core.save('GPU_PREFLIGHT.json',{'time':time.time(),'gpu':gpu,'free_mib':available[gpu]})
        if available[gpu]<45000:raise RuntimeError('Insufficient GPU headroom')
        core.CONTEXT['phase']='loading_six_tools';tools,tool_state,Agent,prompt=core.rt.load_six();core.save('TOOLS_LOADED.json',{'time':time.time(),'names':[t.name for t in tools]})
        DECISION='evaluating_frozen_states'
        for block in range(2):
            items=INPUTS if block==0 else list(reversed(INPUTS))
            for i,item in enumerate(items):
                offset=(i+block)%3
                for arm in ARMS[offset:]+ARMS[:offset]:core.episode(item,STATES[arm],arm+'_r'+str(block+1),tools,tool_state,Agent,prompt,memory,train_ids)
        assert STATES==json.loads((ROOT/'private/states.json').read_text());DECISION='fixed_matrix_complete'
    except BaseException as exc:
        FATAL=repr(exc);core.save('FAILED.json',{'time':time.time(),'error':FATAL,'traceback':traceback.format_exc(),'context':core.CONTEXT})
    finally:finalize()
    print(json.dumps({'finished':True,'fatal':FATAL,'episodes':len(core.EPISODES)}),flush=True)

if __name__=='__main__':main()
