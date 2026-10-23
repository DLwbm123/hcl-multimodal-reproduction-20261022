import asyncio,json,math,time
from pathlib import Path
from adapter import load_upstream,adapter_class
from tools import RestrictedTools
from qwen_transport import LocalModel,RunStopped
from learning import atomic,optimizer,metrics,paired,choose,commit,validate,CANDIDATE_SCHEMA

async def episode(upstream,root,case,model,workflow):
    out=Path(root)/case['id'];out.mkdir(parents=True)
    prompt=out/'prompt.txt';prompt.write_text(case['prompt'])
    args=upstream.parse_args(['--workspace',str(out),'--prompt-file',str(prompt),'--model','frozen-local','--upstream','http://127.0.0.1:1/v1','--planning','on','--context-policy','none','--verification','off','--max-tokens','1024','--context-window','16384','--timeout-seconds','600','--seed','42'])
    tools=RestrictedTools(case['rows']);manifest={}
    h=adapter_class(upstream)(args,'http://127.0.0.1:1/v1',out,upstream.EventLog(out),manifest,tools=tools,model=model,workflow=tuple(workflow))
    await h.setup_tools(None)
    start=model.calls
    final=await h.build_graph().ainvoke({'turn':0,'termination':None,'final_text':None},config={'recursion_limit':110})
    trace=[]
    for event in h.history:
        m=event['message']
        if event['role']=='tool':trace.append({'role':'tool','name':event['tool_name'],'content':m.content,'status':m.status})
        else:trace.append({'role':'assistant','content':m.content,'tool_calls':m.tool_calls,'invalid_tool_calls':m.invalid_tool_calls})
    result={'case_id':case['id'],'termination':final.get('termination'),'answer':tools.answer,'model_calls':model.calls-start,
            'submitted':tools.answer is not None,'trace':trace,'saw_facts':any(x.get('name')=='read_facts' for x in trace),
            'saw_calculation':any(x.get('name')=='calculate' and x.get('status')=='success' for x in trace)}
    if result['termination']=='provider_failure':raise RuntimeError('unclassified provider failure; fatal, no retry')
    return result

def run(root,repo,hcl_path,model_path):
    import sys
    root=Path(root);repo=Path(repo);sys.path[:0]=[str(repo),str(hcl_path)]
    from medharness.scorers.calc import score_prediction
    from hcl.models import LocalTransformersModel
    from langchain_core.messages import HumanMessage,SystemMessage
    upstream=load_upstream(repo)
    visible=json.loads((root/'data/visible_split.json').read_text())
    # Answers remain evaluator-side. Only training feedback is sent to proposer.
    gold=json.loads((root/'data/evaluator_gold.json').read_text())
    assert all('gold' not in c and 'script' not in c for c in visible)
    backend=LocalTransformersModel(name='Qwen2.5-VL-7B-Instruct',path=model_path,torch_dtype='bfloat16',attn_implementation='sdpa',max_new_tokens=1024,answer_max_new_tokens=1024,temperature=0.,enable_thinking=False,local_files_only=True)
    model=LocalModel(backend,root);opt=optimizer(root)
    deployed={'version':0,'sections':[]};atomic(root/'deployed.json',deployed)
    atomic(root/'effective_optimizer.json',{k:getattr(opt,k) for k in ['candidate_count','components','historical_loss_budget','min_correct_gain','min_primary_score_delta','min_format_compliance_rate','min_historical_anchor_count','min_historical_anchors_per_task','min_anchor_evaluation_coverage','screening_ratio','selection_mode']})
    all_records=[];events=[]
    def select(*roles):return [c for c in visible if c['role'] in roles]
    def evaluate(cases,sections,label):
        saved=json.loads((root/'deployed.json').read_text());records=[]
        for case in cases:
            atomic(root/'progress.json',{'phase':label,'case':case['id'],'model_calls':model.calls,'time':time.time()})
            result=asyncio.run(episode(upstream,root/'episodes'/label,case,model,sections))
            score=score_prediction(result['answer'] or {'calculators':[]},gold[case['id']])
            result.update(utility=score['utility'],success=bool(result['submitted'] and score['utility']==1 and result['saw_facts'] and result['saw_calculation']),score=score)
            atomic(root/'episodes'/label/case['id']/'RESULT.json',result)
            records.append(result)
            all_records.append({'phase':label,**{k:v for k,v in result.items() if k not in ('trace','answer','score')}})
            with (root/'evaluations.jsonl').open('a') as f:f.write(json.dumps(all_records[-1])+'\n')
            assert json.loads((root/'deployed.json').read_text())==saved,'evaluation mutated deployed state'
        return records
    for stage,task in enumerate(['dependency','submission'],1):
        train=select('train_'+task);selection=select('select_'+task)
        anchors=select('anchor_basic')+(select('anchor_dependency') if stage==2 else [])
        training=evaluate(train,deployed['sections'],f's{stage}_train')
        # Complete target and historical baseline before any candidate can commit.
        current=evaluate(selection,deployed['sections'],f's{stage}_current')
        historical=evaluate(anchors,deployed['sections'],f's{stage}_anchors')
        feedback=[{'prompt':c['prompt'],'trajectory':[{k:(str(v)[:500] if k=='content' else v) for k,v in e.items()} for e in r['trace'] if e.get('name')!='read_facts'][-12:],'submitted_answer':r['answer'],
                   'utility':r['utility'],'correct_feedback':gold[c['id']]} for c,r in zip(train,training)]
        generation_context={'current_workflow':deployed['sections'],'training':feedback}
        atomic(root/f'generation_input_s{stage}.json',generation_context)
        candidates=[]
        for index in range(2):
            request=[SystemMessage(content='You propose reusable workflow instructions for a frozen tool-using agent. Learn only from the supplied training trajectories and feedback. Change workflow guidance only. Do not change tools, schemas, scoring, permissions, model weights, or budgets. Do not include case-specific facts, numbers, answers, executable code, placeholders, or identifiers. Return one short candidate via propose_workflow. Each instruction must be procedural prose. No hand-written solution is supplied.'),
                     HumanMessage(content=json.dumps({'proposal_index':index,'previous_proposals':[c['artifact_content'] for c in candidates],**generation_context}))]
            model.bind_tools([CANDIDATE_SCHEMA])
            try:
                response=asyncio.run(model.ainvoke(request))
            except (ValueError,KeyError,TypeError) as exc:
                atomic(root/f'proposal_s{stage}_{index}.json',{'rejected':True,'error':str(exc),'failure_class':getattr(model,'last_failure',None)})
                continue
            calls=response.tool_calls
            raw=[calls[0]['args']] if len(calls)==1 and calls[0]['name']=='propose_workflow' and not response.invalid_tool_calls else []
            normalized,rejections=opt._normalize_candidates(raw,component='medical_workflow',source_version=deployed['version'],candidate_version=stage*10+index,count=1,current_artifact={'content':{'sections':deployed['sections']}})
            for c in normalized:c['validation_errors']+=validate(c['artifact_content']['sections'])
            candidates+=normalized
            atomic(root/f'proposal_s{stage}_{index}.json',{'raw':response.model_dump(mode='json'),'candidates':normalized,'rejections':rejections})
        opt.record_candidates(candidates)
        evaluations={}
        for i,c in enumerate(candidates):
            if c['validation_errors']:continue
            # Reserve the worst-case 24 episodes x 24 calls for final paired evaluation.
            remaining=json.loads((root/'budget.json').read_text())['max_model_calls']-model.calls
            if remaining<576+24*(len(selection)+len(anchors)):
                raise RunStopped('insufficient budget for complete candidate plus independent confirmation')
            sections=c['artifact_content']['sections']
            evaluations[c['candidate_id']]={'current':evaluate(selection,sections,f's{stage}_candidate{i}_current'),
                                           'anchors':evaluate(anchors,sections,f's{stage}_candidate{i}_anchors')}
        chosen=choose(opt,candidates,evaluations,current,historical)
        before=dict(deployed);deployed=commit(root/'deployed.json',deployed,chosen,stage)
        event={'stage':stage,'before':before,'after':deployed,'decision':opt.history[-1],
               'baseline_current':metrics(current),'historical_success_count':sum(r['success'] for r in historical),
               'candidate_pairs':{k:paired(historical,v['anchors']) for k,v in evaluations.items()}}
        events.append(event);atomic(root/f'stage{stage}.json',event)
    confirmation=select('confirm_basic','confirm_dependency','confirm_submission')
    assert len(confirmation)==12
    # This is the first execution of confirmation cases. Results never select state.
    static=evaluate(confirmation,[], 'final_static')
    learned=evaluate(confirmation,deployed['sections'],'final_hcl')
    delta=paired(static,learned)
    import random
    rng=random.Random(2026102305);diff=[int(b['success'])-int(a['success']) for a,b in zip(static,learned)]
    boot=sorted(sum(rng.choices(diff,k=12))/12 for _ in range(20000))
    n=delta['forget_count']+delta['recovered_count'];k=min(delta['forget_count'],delta['recovered_count'])
    p=min(1.,2*sum(math.comb(n,j) for j in range(k+1))/2**n) if n else 1.
    family={role:{'static':metrics([r for r in static if r['case_id'].startswith(role)]),'hcl':metrics([r for r in learned if r['case_id'].startswith(role)])} for role in ['confirm_basic','confirm_dependency','confirm_submission']}
    result={'complete':True,'stages':events,'final_state':deployed,'static':metrics(static),'hcl':metrics(learned),'paired':delta,'ci95':[boot[500],boot[19500]],'mcnemar_p':p,'families':family,'model_calls':model.calls,
      'mechanism_signal':all(e['historical_success_count']>0 for e in events) and deployed['version']>0 and delta['recovered_count']>delta['forget_count'] and delta['forget_count']==0,
      'scope':'synthetic restricted workflow-only HCL; not clinical or full original reproduction'}
    atomic(root/'RESULT.json',result)
    return 0
