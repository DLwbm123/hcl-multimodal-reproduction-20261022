"""Training-feedback query executes jointly on all current images before native reasoning."""
import json,os,signal,subprocess,threading,time,traceback
from pathlib import Path
import run as core
from logic import paired
from storage import tree_bytes
from query_logic import ARMS,FINAL,FIXED_QUERY,TOOLS,parse,query,arguments,prefix_text,decision
ROOT=core.ROOT;BUDGET=core.BUDGET
INPUTS=[];STATES={};FATAL=None;POLICY_ACCEPTED=False
os.environ['TMPDIR']='/data/bmw/workers/t1018';os.environ['TRITON_CACHE_DIR']='/data/bmw/workers/t1018/triton'


def watchdog():
    names=['medrax_subscription_v4','medrax_hcl_v5','medrax_memory_interface_v6','medrax_corrective_memory_v7','medrax_error_abstraction_v8','medrax_learning_qualification_v9','medrax_evidence_memory_v10','medrax_visual_memory_v11','medrax_postdraft_memory_v12','medrax_tool_query_v13','medrax_autonomy_12h_20261025'];roots=[ROOT]+[ROOT.parent/n for n in names]
    while True:
        try:
            sizes=[tree_bytes(p) for p in roots];core.save('RUN_GUARD.json',{'time':time.time(),'own_bytes':sizes[0],'aggregate_bytes':sum(sizes),'deadline':BUDGET['hard_deadline_epoch']})
            if sizes[0]>=BUDGET['own_stop_new_bytes'] or sum(sizes)>=BUDGET['aggregate_stop_new_bytes'] or time.time()>=BUDGET['hard_deadline_epoch']:raise RuntimeError('Original time or storage stop')
        except BaseException as exc:
            try:core.save('STOPPED.json',{'time':time.time(),'error':repr(exc),'context':core.CONTEXT})
            finally:os.killpg(os.getpgrp(),signal.SIGTERM)
            return
        time.sleep(15)


def propose():
    from langchain_core.messages import HumanMessage,SystemMessage
    core.guard();core.CONTEXT.update(phase='policy_generation',episode=None,stage='B',state='single_query_proposal')
    instruction='Generate one restricted reusable tool-query program from historical training feedback. The source benchmark targets were NOT independently clinically adjudicated. Learn a procedure for eliciting observable current-image facts; do not convert uncertain target diagnoses into medical rules. No target evaluation examples are provided. Do not call tools. Return only one JSON object with exactly admit(boolean), tool_name, query_template, source_indices, rationale. If no defensible procedural proposal is possible, admit=false and explain, using empty tool_name/query_template/source_indices. If admitted, tool_name must be chest_xray_expert; query_template must be250-1600 characters and contain exactly one literal {question_stem} placeholder and no other braces; source_indices must be4-24 distinct integers0-23. The program receives ALL2-4 current-case images TOGETHER in their supplied order in ONE chest_xray_expert call. Learn a procedure to compare visible evidence across these supplied views or panels. Do not assume temporal order, treatment response, clinical outcome, a diagnosis or an answer letter absent from explicit question context. Do not convert symptoms into image findings. Ask for visible facts relevant to the supplied question context and state limitations. The neutral control uses the SAME chosen tool and one JOINT call on the SAME2-4 images with the fixed generic query supplied below. Do not merely copy that control verbatim. There is exactly one proposal opportunity, no repair or regeneration.'
    training=json.loads((ROOT/'private/source_training.json').read_text());payload={'fixed_generic_query':FIXED_QUERY,'source_training':training,'note':'Source option texts are unadjudicated benchmark supervision. No target gold, target question, historical target result or image enters this proposal.'}
    model=core.rt.SubscriptionModel();model.deadline=min(time.time()+1200,BUDGET['hard_deadline_epoch']);out=model.invoke([SystemMessage(content=instruction),HumanMessage(content=json.dumps(payload,ensure_ascii=False))]);raw={'time':time.time(),'content':out.content,'tool_calls':out.tool_calls,'model_calls':model.calls};core.save('private/policy_generation.json',raw);assert not out.tool_calls,'Unexpected proposal tool call'
    value=parse(out.content);core.save('private/query_candidate.json',value)
    if not value['admit']:raise RuntimeError('Single query proposal declined; no evaluation')
    states={'H0_static':{'policy':None},'fixed_joint':{'policy':{'tool_name':value['tool_name'],'query_template':FIXED_QUERY}},'learned_joint':{'policy':value}}
    core.save('private/states.json',states);core.save('STATES_FROZEN.json',{'time':time.time(),'one_proposal_admitted':True,'source_cases':24,'tool_name':value['tool_name'],'supporting_source_count':len(value['source_indices']),'target_inputs_not_used_for_proposal':True,'medical_validation':False});return states


def execute_prefix(item,policy,tools):
    if policy is None:return []
    by={t.name:t for t in tools};text=query(policy,item['question']);rows=[]
    core.guard();result=by[policy['tool_name']].invoke(arguments(policy['tool_name'],item['images'],text));status=result[1].get('analysis_status','unknown') if isinstance(result,tuple) else 'unknown';value=result[0] if isinstance(result,tuple) else result
    if isinstance(value,dict) and 'response' in value:value=value['response']
    rows.append({'image_indices':list(range(1,len(item['images'])+1)),'tool_name':policy['tool_name'],'query':text,'status':status,'observation':str(value).strip()[:12000]})
    return rows


def episode(item,arm,block,tools,tool_state,Agent,prompt):
    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.memory import MemorySaver
    core.guard();label=arm+'_r'+str(block);core.CONTEXT.update(phase='case',episode=len(core.EPISODES),stage='B',state=label)
    before=json.dumps(STATES[arm],sort_keys=True);paths=item['images'];tool_state.update(calls=0,success=0,images=set(paths));started=time.time();outdir=ROOT/'outputs'/str(len(core.EPISODES));outdir.mkdir(exist_ok=False)
    result={'id':item['id'],'stage':'B','role':'adaptive_error_enriched_development','state':label,'status':'failed','predicted':None,'prefix_text':'','prefix_calls':0};model=None
    try:
        observations=execute_prefix(item,STATES[arm]['policy'],tools);text=prefix_text(observations);core.save('private/prefix_'+str(len(core.EPISODES))+'.json',observations);result.update(prefix_text=text,prefix_calls=len(observations),prefix_tool_name=STATES[arm]['policy']['tool_name'] if observations else None)
        model=core.rt.SubscriptionModel();model.deadline=min(started+core.PLAN['limits']['per_case_seconds'],BUDGET['hard_deadline_epoch']);agent=Agent(model,tools,system_prompt=prompt,checkpointer=MemorySaver(),log_dir=str(outdir));config={'configurable':{'thread_id':'1'},'recursion_limit':2*core.PLAN['limits']['per_case_controller_calls']+2}
        figure=''.join(Path(p).name+' located at '+p+'\n' for p in paths);user='Answer this question correctly using chain of thought reasoning and carefully evaluating choices. Solve using our own vision and reasoning and thenuse tools to complement your reasoning. Trust your own judgement over any tools.\n'+item['question']+'\n'+figure+text;content=[{'type':'text','text':user}]+[{'type':'image_url','image_url':{'url':p}} for p in paths]
        out=agent.workflow.invoke({'messages':[HumanMessage(content=content)]},config);result['initial_final']=out['messages'][-1].content;out=agent.workflow.invoke({'messages':[HumanMessage(content=FINAL)]},config);answer=out['messages'][-1].content.strip();result.update(status='completed',final=answer,predicted=answer if answer in list('ABCDEF') else None)
    except Exception as exc:result['error']=repr(exc);core.record('episode_error',{'error':repr(exc)})
    assert json.dumps(STATES[arm],sort_keys=True)==before
    result.update(seconds=time.time()-started,controller_calls=model.calls if model else 0,tool_calls=tool_state['calls'],successful_tools=tool_state['success']);core.EPISODES.append(result);core.save('private/episodes.json',core.EPISODES);core.save('PROGRESS.json',{'time':time.time(),'episodes':len(core.EPISODES),'context':core.CONTEXT,'last_status':result['status']});core.guard()


def finalize():
    gold=json.loads((ROOT/'private/sealed_development_gold.json').read_text());ids=[x['id'] for x in INPUTS];by={};metrics={};pairs=[]
    for b in [1,2]:
        for arm in ARMS:
            label=arm+'_r'+str(b);actual={r['id']:r for r in core.EPISODES if r['state']==label};by[label]={k:dict(actual.get(k,{'id':k,'status':'not_run','predicted':None})) for k in ids};metrics[label]={'denominator':len(ids),'correct':sum(r.get('predicted')==gold[k] for k,r in by[label].items()),'valid':sum(r.get('status')=='completed' and r.get('predicted') in list('ABCDEF') for r in actual.values()),'prefix_calls':sum(r['prefix_calls'] for r in actual.values())}
        get=lambda a:list(by[a+'_r'+str(b)].values());pairs.append({k:paired(get(a),get('learned_joint'),ids,gold) for k,a in [('H0_to_learned','H0_static'),('fixed_to_learned','fixed_joint')]})
    complete=FATAL is None and POLICY_ACCEPTED and len(core.EPISODES)==36 and all(v['valid']==6 for v in metrics.values());g=decision(pairs,by,ids,gold,complete);core.save('private/gate.json',g);g={k:v for k,v in g.items() if k!='repeat_observed_gain_ids'}
    report={'time':time.time(),'fatal':FATAL,'complete':complete,'policy_proposal_admitted':POLICY_ACCEPTED,'planned_episodes':36,'physical_episodes':len(core.EPISODES),'metrics':metrics,'paired_per_repeat':pairs,'acquisition_gate':g,'error_enriched_reused_B_cases':6,'completed_policy_proposal_calls':1 if (ROOT/'private/policy_generation.json').exists() else 0,'confirmation_used':False,'medical_validity_established':False,'sequential_learning_established':False,'seconds_since_original_start':time.time()-BUDGET['started_epoch']};core.save('private/results.json',by);core.save('AGGREGATE_REPORT.json',report);core.save('FINISHED.json',report)


def main():
    global INPUTS,STATES,FATAL,POLICY_ACCEPTED
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('Independent watchdog termination')));threading.Thread(target=watchdog,daemon=True).start();core.save('STARTED.json',{'time':time.time(),'original_start':BUDGET['started_epoch'],'deadline':BUDGET['hard_deadline_epoch']});INPUTS=json.loads((ROOT/'private/inputs.json').read_text())
    try:
        STATES=propose();POLICY_ACCEPTED=True
        s=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True);free={int(x.split(',')[0]):int(x.split(',')[1]) for x in s.strip().splitlines()};gpu=int(os.environ['CUDA_VISIBLE_DEVICES']);core.save('GPU_PREFLIGHT.json',{'time':time.time(),'gpu':gpu,'free_mib':free[gpu]})
        if free[gpu]<45000:raise RuntimeError('Insufficient GPU headroom before tool load')
        tools,state,Agent,prompt=core.rt.load_six();core.save('TOOLS_LOADED.json',{'time':time.time(),'names':[t.name for t in tools]})
        for b in [1,2]:
            for i,item in enumerate(INPUTS if b==1 else reversed(INPUTS)):
                offset=(i+b-1)%3
                for arm in ARMS[offset:]+ARMS[:offset]:episode(item,arm,b,tools,state,Agent,prompt)
        assert STATES==json.loads((ROOT/'private/states.json').read_text())
    except BaseException as exc:FATAL=repr(exc);core.save('FAILED.json',{'time':time.time(),'error':FATAL,'traceback':traceback.format_exc(),'context':core.CONTEXT})
    finally:finalize()
    print(json.dumps({'finished':True,'fatal':FATAL,'episodes':len(core.EPISODES)}),flush=True)

if __name__=='__main__':main()
