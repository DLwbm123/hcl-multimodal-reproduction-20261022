"""One native medical Agent A→B run using existing HCL components.

This adapter does not claim to execute HCLPipeline's dataset-oriented driver.
It reuses TaskInterface, Router, ExperienceMemory and Optimizer, and supplies
native Agent evaluation with an explicit binary retention gate.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

import base_worker as base
import run_agent as rt
from policy import FIXED
from qualified_tool import install_shared_guard
from state_store import StateStore, gate, paired, state_diff, write_json
from hcl.memory import ExperienceMemory, MemoryConfig
from hcl.optimizer import Optimizer
from hcl.router import Router, RouterConfig, DEFAULT_MEMORY_SELECTOR_TEMPLATE
from hcl.task_interface import TaskInterface, TaskInterfaceConfig

ROOT, PLAN, BUDGET = rt.ROOT, rt.PLAN, rt.BUDGET
CONTEXT, EVENTS, EPISODES = base.CONTEXT, base.EVENTS, base.EPISODES
save, record, guard = base.save, base.record, base.guard
STATES, DECISIONS, PROPOSALS, REPORT_ROWS = {}, {}, {}, []
ADAPTER_CALLS = []
READ_ONLY_CHECKS = 0
CURRENT_EXTRA = ''
base.policy_context = lambda *_: (CURRENT_EXTRA, [])


class Controller:
    def __init__(self):
        self.training_evidence = []

    def generate(self, prompt, *, state):
        from langchain_core.messages import HumanMessage, SystemMessage
        guard()
        model = rt.SubscriptionModel()
        model.deadline = min(time.time() + 300, BUDGET['hard_deadline_epoch'])
        safety = ('Follow the requested JSON schema. Return content as JSON and no tool calls. '
                  'Historical benchmark feedback is unadjudicated supervision, not clinical truth. '
                  'Propose transferable process guidance, not new diagnostic claims, patient facts, '
                  'answers for unseen cases, or executable code. Never change labels, scoring, tools, '
                  'privacy guards or budgets. Do not manufacture success. Returning no justified skill is valid.')
        if self.training_evidence and CONTEXT.get('phase') == 'proposal':
            safety += '\nObserved training execution only:\n' + json.dumps(self.training_evidence)
        out = model.invoke([SystemMessage(content=safety), HumanMessage(content=prompt)])
        if out.tool_calls:
            raise ValueError('HCL metadata generation must not execute tools')
        entry = {'time': time.time(), **CONTEXT, 'request_state': state,
                 'prompt': prompt, 'output': out.content, 'controller_calls': model.calls}
        ADAPTER_CALLS.append(entry)
        save('private/hcl_calls.json', ADAPTER_CALLS)
        return out.content


def native_context(snapshot, item, memory, model):
    """Only current visible input and training-derived memory reach selection."""
    memory.restore(snapshot['memory'])
    example = {'task_id': item['id'], 'question': item['question'],
               'image_paths': item['images'], 'visible_context': {'image_count': len(item['images'])}}
    interface = TaskInterface(TaskInterfaceConfig(mode='rule'))
    chunk = interface.build_taskinterfacechunk(example, 0)
    if not any(snapshot['memory'][k] for k in ['raw', 'abstract', 'skills']):
        record('hcl_route', {'interface': TaskInterface.to_model_visible(chunk),
                            'selected_memory': [], 'selected_skills': [], 'reason': 'empty_initial_memory'})
        return snapshot['known_prior']
    config = RouterConfig(use_llm_workflow_selector=False, use_tool_selector=False,
                          use_memory_selector=True, use_skill_selector=True,
                          memory_selector_invalid_output_retries=1,
                          memory_selector_fallback_to_empty_on_invalid=False)
    router = Router(model, config, memory=memory,
                    memory_selector_template_content=snapshot['memory_selector'],
                    context_template_content=snapshot['context_template'])
    skills = router.select_skills(chunk)
    workflow = router.plan_workflow(chunk, selected_skills=skills)
    selected, trace = router.select_memory(chunk, workflow)
    # The native Agent selects and invokes real tools; core BasicCapability tools are disabled.
    context = router.build_context(chunk, workflow_decision=workflow,
                                   selected_memory=selected, selected_skills=skills, selected_tools=[])
    record('hcl_route', {'interface': TaskInterface.to_model_visible(chunk), 'workflow': workflow,
                        'selection': trace, 'selected_memory': selected, 'selected_skills': skills})
    return snapshot['known_prior'] + '\n' + context


def run_one(item, snapshot, label, memory, model, native):
    global CURRENT_EXTRA, READ_ONLY_CHECKS
    tools, tool_state, Agent, prompt = native
    guard()
    CONTEXT.update(phase='episode', stage=item['stage'], role=item['role'],
                   episode_id=item['id'], arm=label)
    model.training_evidence = []
    frozen = deepcopy(snapshot)
    start_events, start_calls, start_episodes = len(EVENTS), len(ADAPTER_CALLS), len(EPISODES)
    started = time.time()
    try:
        CURRENT_EXTRA = '' if label == 'H0' else native_context(snapshot, item, memory, model)
        if label == 'H0':
            memory.restore(snapshot['memory'])
        extra = len(ADAPTER_CALLS) - start_calls
        if extra > PLAN['limits']['max_router_calls_per_case']:
            raise RuntimeError('Per-case HCL routing budget exceeded')
        base.episode(item, label, tools, tool_state, Agent, prompt, None, None)
        result = EPISODES[-1]
    except Exception as exc:
        if len(EPISODES) != start_episodes:
            raise
        result = {'id': item['id'], 'arm': label, 'status': 'failed', 'predicted': None,
                  'error': repr(exc), 'reference_tier': item['reference_tier'],
                  'controller_calls': 0, 'tool_calls': 0, 'successful_tools': 0}
        EPISODES.append(result)
        record('integration_episode_failure', {'error': repr(exc)})
    if memory.snapshot() != snapshot['memory'] or snapshot != frozen:
        raise RuntimeError('Evaluation mutated persistent HCL state')
    READ_ONLY_CHECKS += 1
    result.update(stage=item['stage'], role=item['role'], seconds=time.time()-started,
                  hcl_calls=len(ADAPTER_CALLS)-start_calls,
                  all_controller_calls=sum(e['kind']=='controller_finished' for e in EVENTS[start_events:]))
    save('private/episodes.json', EPISODES)
    save('PROGRESS.json', {'time': time.time(), 'recorded_episodes': len(EPISODES),
                          'context': CONTEXT, 'read_only_checks': READ_ONLY_CHECKS})
    return result


def proposals(before, items, predictions, train_gold, stage, memory, model):
    CONTEXT.update(phase='proposal', stage=stage, role='train', arm='candidate_generation')
    model.training_evidence = [
        {'status': p['status'], 'benchmark_agreement': p.get('predicted')==train_gold[x['id']],
         'tool_calls': p.get('tool_calls'), 'successful_tools': p.get('successful_tools'),
         'error': p.get('error'), 'reference_status': x['reference_tier'],
         'execution_trace': [{k:e[k] for k in ['kind','name','status','args','result'] if k in e}
                             for e in EVENTS if e.get('episode_id')==x['id']
                             and e.get('arm')==stage+'_train'
                             and e['kind'] in ['tool_started','tool_finished','episode_error']]}
        for x, p in zip(items, predictions)]
    memory.restore(before['memory'])
    examples = []
    for item, pred in zip(items, predictions):
        example = {'task_id': item['id'], 'task_name': stage, 'task_type': 'medical_agent',
                   'question': item['question'], 'answer': train_gold[item['id']],
                   'image_path': item['images'][0], 'split': 'train'}
        examples.append(example)
        memory.record_training_interaction(stage, example,
                                           {'task_id': item['id'], 'answer': pred.get('predicted')})
    update = memory.consolidate_task(stage, examples)
    candidate = deepcopy(before)
    candidate['memory'] = memory.snapshot()
    candidate['version'] = before['version'] + 1
    assert not candidate['memory']['anchor']
    options = [('memory', candidate)]
    optimizer = Optimizer(record_dir=ROOT/'private/optimizer'/stage, candidate_count=1,
                          components=['memory_selector'], historical_loss_budget=0,
                          min_correct_gain=1, min_primary_score_delta=0.25,
                          min_historical_anchor_count=2, min_historical_anchors_per_task=2,
                          screening_min_current_count=4, screening_min_anchors_per_task=2)
    # Selection and anchor examples/answers never enter this call.
    generated = optimizer.generate_artifact_candidates(
        'memory_selector', {'metrics': {'training_count': len(items),
                                       'primary_score': sum(p.get('predicted')==train_gold[x['id']]
                                                            for x,p in zip(items,predictions))/len(items)}},
        model=model, current_artifact={'content': before['memory_selector']},
        harness_version=before['version'],
        generation_context={'training_execution': model.training_evidence,
                            'training_abstract_memory': candidate['memory']['abstract'],
                            'fixed_boundary': 'Select useful historical process guidance or nothing. '
                            'No current reference answer, no task/stage oracle, no clinical-truth claim.'})
    valid = [c for c in generated if not c['validation_errors']]
    if valid and valid[0]['artifact_content'] != before['memory_selector']:
        second = deepcopy(candidate)
        second['memory_selector'] = valid[0]['artifact_content']
        options.append(('memory_plus_selector', second))
    PROPOSALS[stage] = {'memory_update': update, 'generated_selector_count': len(generated),
                        'valid_selector_count': len(valid), 'candidate_names': [n for n,_ in options],
                        'selector_diagnostics': optimizer.last_generation_diagnostics}
    memory.restore(before['memory'])
    model.training_evidence = []
    save('private/proposals.json', PROPOSALS)
    return options


def report(inputs, report_gold, aliases):
    metrics, pairs = {}, {}
    for name in ['H0', 'Fixed', 'S0', 'SA', 'SAB']:
        rows = [r for r in REPORT_ROWS if r['arm']==aliases.get(name, name)]
        metrics[name] = {}
        for stage in ['A', 'B']:
            ids = [x['id'] for x in inputs if x['role']=='report' and x['stage']==stage]
            by = {r['id']:r for r in rows}
            metrics[name][stage] = {'denominator':len(ids),
                'correct':sum(by.get(k,{}).get('predicted')==report_gold[k] for k in ids),
                'completed':sum(by.get(k,{}).get('status')=='completed' for k in ids),
                'valid_final':sum(by.get(k,{}).get('predicted') in list('ABCDEF') for k in ids)}
    def rows(name):return [r for r in REPORT_ROWS if r['arm']==aliases.get(name,name)]
    for a,b in [('H0','SAB'),('Fixed','SAB'),('S0','SA'),('SA','SAB')]:
        pairs[a+'_to_'+b]={s:paired(rows(a),rows(b),
            [x['id'] for x in inputs if x['role']=='report' and x['stage']==s],report_gold)
            for s in ['A','B']}
    a_ids=[x['id'] for x in inputs if x['role']=='report' and x['stage']=='A']
    def correct(name):return {r['id'] for r in rows(name) if r['id'] in a_ids and r.get('predicted')==report_gold[r['id']]}
    acquired=correct('SA')-correct('S0')
    retention={'acquired_A_count':len(acquired), 'acquired_A_lost':len(acquired-correct('SAB')) if acquired else None,
               'acquired_A_retention':(len(acquired & correct('SAB'))/len(acquired)) if acquired else None,
               'base_A_count':len(correct('S0')),'base_A_lost':len(correct('S0')-correct('SAB'))}
    return {'metrics':metrics,'paired':pairs,'retention':retention,
            'state_aliases':aliases,'development_replay_only':True,'clinical_correctness_established':False}


def main():
    global CURRENT_EXTRA
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(TimeoutError('Independent supervisor stop')))
    inputs=json.loads((ROOT/'private/inputs.json').read_text())
    from preflight import check_inputs
    preflight=check_inputs(ROOT, inputs)
    save('PREFLIGHT.json',preflight)
    save('STARTED.json',{'time':time.time(),'original_start':BUDGET['started_epoch'],
                         'deadline':BUDGET['hard_deadline_epoch'],'gpu':int(os.environ['CUDA_VISIBLE_DEVICES'])})
    store=StateStore(ROOT/'private/states')
    model=Controller()
    memory=ExperienceMemory(MemoryConfig(record_dir=str(ROOT/'private/transient_memory'),
        raw_capacity_per_task=8, abstract_mode='llm', abstract_prompt_max_chars=16000,
        skill_enabled=True, skill_min_source_tasks=2, anchor_selection_mode='off'),model=model)
    selector=json.loads(DEFAULT_MEMORY_SELECTOR_TEMPLATE.read_text())
    selector['sections']=[s.replace('correct_feedback is authoritative;',
        'correct_feedback is historical benchmark supervision, not established clinical truth;')
        for s in selector['sections']]
    s0={'version':0,'memory':memory.snapshot(),'memory_selector':selector,'known_prior':FIXED,
        'context_template':{'template_id':'medical.context.fixed','sections':[
          'Use historical guidance only when applicable; it is not evidence about the current patient.',
          'Historical correct_feedback denotes unadjudicated benchmark supervision; previous_attempt may be incorrect.',
          'Preserve current image/call provenance and all shared tool limits. Never infer findings from failed tools.',
          'Current visible task: {model_visible_input}', 'Workflow: {workflow_decision}',
          'Selected memory: {selected_memory}', 'Selected skills: {selected_skills}',
          'Native tool choice remains with the real controller. {selected_tools}']},
        'fixed_components':{'interface':'TaskInterface rule','workflow':'native MedRAX graph',
                            'capabilities':'six native tools plus known focused prior; no new tool code'}}
    STATES['S0']=store.deploy('S0',s0,None,{'initial':True})
    fatal=None;aliases={'S0':'Fixed'}
    try:
        available={int(x.split(',')[0]):int(x.split(',')[1]) for x in subprocess.check_output(
            ['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()}
        gpu=int(os.environ['CUDA_VISIBLE_DEVICES']);save('GPU_PREFLIGHT.json',{'gpu':gpu,'free_mib':available[gpu]})
        if available[gpu]<45000:raise RuntimeError('Insufficient GPU headroom')
        CONTEXT['phase']='loading_six_tools';native=rt.load_six()
        install_shared_guard(native[0],native[1],record)
        save('TOOLS_LOADED.json',{'time':time.time(),'names':[t.name for t in native[0]]})
        def subset(stage,role):return [x for x in inputs if x['stage']==stage and x['role']==role]
        train_gold=json.loads((ROOT/'private/train_gold.json').read_text())
        selection_gold=json.loads((ROOT/'private/selection_gold.json').read_text())
        anchor_gold=json.loads((ROOT/'private/anchor_gold.json').read_text())
        for stage,outname,parent in [('A','SA','S0'),('B','SAB','SA')]:
            before=store.load()
            assert before==STATES[parent]
            train=subset(stage,'train');selection=subset(stage,'selection')
            anchors=subset('A','anchor') if stage=='B' else []
            pred=[run_one(x,before,stage+'_train',memory,model,native) for x in train]
            options=proposals(before,train,pred,train_gold,stage,memory,model)
            reference=[run_one(x,before,stage+'_reference',memory,model,native) for x in selection]
            ar=[run_one(x,before,stage+'_anchor_reference',memory,model,native) for x in anchors]
            evaluated=[]
            for name,candidate in options:
                store.save(stage+'_'+name+'_candidate',candidate)
                selected=[run_one(x,candidate,stage+'_'+name,memory,model,native) for x in selection]
                ap=[run_one(x,candidate,stage+'_'+name+'_anchor',memory,model,native) for x in anchors]
                decision=gate(reference,selected,selection_gold,[x['id'] for x in selection],
                              ar,ap,anchor_gold,[x['id'] for x in anchors])
                decision.update(candidate=name,diff=state_diff(before,candidate))
                evaluated.append((decision,candidate))
                # Candidate execution cannot leave transient state deployed after rejection/crash.
                memory.restore(store.load()['memory'])
                assert memory.snapshot()==before['memory'] and store.load()==before
                decision['restored_incoming_after_evaluation']=True
                save('private/decision_'+stage+'_'+name+'.json',decision)
            accepted=[(d,c) for d,c in evaluated if d['accept']]
            chosen=max(accepted,key=lambda dc:dc[0]['after_correct']) if accepted else None
            after=deepcopy(chosen[1] if chosen else before)
            DECISIONS[stage]={'candidates':[d for d,_ in evaluated],
                              'selected':chosen[0]['candidate'] if chosen else None,
                              'accepted':chosen is not None,'diff':state_diff(before,after)}
            STATES[outname]=store.deploy(outname,after,parent,DECISIONS[stage])
            memory.restore(STATES[outname]['memory'])
            save('DECISIONS.json',DECISIONS)
        # All states fixed before any report prediction or report label access.
        unique={'Fixed':STATES['S0']}
        for name in ['SA','SAB']:
            same=next((key for key,v in unique.items() if v==STATES[name]),None)
            aliases[name]=same or name
            if same is None:unique[name]=STATES[name]
        save('STATES_FROZEN.json',{'time':time.time(),'aliases':aliases,'report_gold_read':False})
        names=['H0']+list(unique)
        for i,item in enumerate(x for x in inputs if x['role']=='report'):
            offset=i%len(names)
            for name in names[offset:]+names[:offset]:
                REPORT_ROWS.append(run_one(item,STATES['S0'] if name=='H0' else unique[name],name,memory,model,native))
        memory.restore(store.load()['memory'])
    except BaseException as exc:
        fatal=repr(exc);save('FAILED.json',{'time':time.time(),'error':fatal,'traceback':traceback.format_exc(),'context':CONTEXT})
        memory.restore(store.load()['memory'])
    finally:
        gold=json.loads((ROOT/'private/report_gold.json').read_text())
        summary=report(inputs,gold,aliases)
        valid_report=all(x['completed']==x['denominator'] and x['valid_final']==x['denominator']
                         for v in summary['metrics'].values() for x in v.values())
        loop_complete=(fatal is None and set(DECISIONS)=={'A','B'}
                       and all(d['candidates'] for d in DECISIONS.values()) and valid_report)
        summary.update(fatal=fatal,engineering_loop_complete=loop_complete,decisions=DECISIONS,
                       native_episodes=len(EPISODES),hcl_metadata_calls=len(ADAPTER_CALLS),
                       controller_calls=sum(e['kind']=='controller_finished' for e in EVENTS),
                       tool_calls=sum(e['kind']=='tool_started' for e in EVENTS),read_only_checks=READ_ONLY_CHECKS,
                       accepted_updates=sum(d['accepted'] for d in DECISIONS.values()),
                       finished_at=time.time(),seconds_since_original_start=time.time()-BUDGET['started_epoch'])
        save('private/report_rows.json',REPORT_ROWS)
        save('AGGREGATE_REPORT.json',summary);save('FINISHED.json',summary)
    return 0 if fatal is None else 2


if __name__=='__main__':
    raise SystemExit(main())
