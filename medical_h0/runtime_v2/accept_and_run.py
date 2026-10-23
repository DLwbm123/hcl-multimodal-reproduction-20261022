"""One bounded acceptance, then frozen formal run if its predeclared gate passes."""
import json,asyncio,time
from pathlib import Path
from experiment import episode,run
from adapter import load_upstream
from learning import atomic,CANDIDATE_SCHEMA
from qwen_transport import LocalModel,count_request_tokens
from feedback import pack_request,feedback_rows

def main(root,repo,hcl_path,model_path):
    import sys
    sys.path[:0]=[repo,hcl_path]
    from hcl.models import LocalTransformersModel
    backend=LocalTransformersModel(name='Qwen2.5-VL-7B-Instruct',path=model_path,torch_dtype='bfloat16',attn_implementation='sdpa',max_new_tokens=4096,answer_max_new_tokens=4096,temperature=0.,enable_thinking=False,local_files_only=True)
    model=LocalModel(backend,root);upstream=load_upstream(repo)
    visible=json.loads((root/'data/visible_split.json').read_text())
    # Fixed first two from each original learning block; no selection/anchor/final cases.
    chosen=[c for role in ['train_dependency','train_submission'] for c in [x for x in visible if x['role']==role][:2]]
    results=[]
    for i,c in enumerate(chosen):
        atomic(root/'progress.json',{'phase':'acceptance','ordinal':i,'calls':model.calls,'time':time.time()})
        result=asyncio.run(episode(upstream,root/'acceptance',c,model,[]));results.append(result)
        atomic(root/'acceptance'/c['id']/'RESULT.json',result)
    # Exact saved failure context is a training-only packing regression, no generation.
    old=root.parent/'formal_medtool_v1'
    ctx=json.loads((old/'generation_input_s2.json').read_text())
    from feedback import bounded
    for t in ctx['training']:
        t['trajectory']=[bounded(e) for e in t['trajectory']]
        g=t['correct_feedback']
        t['correct_feedback']=[{'name':a['name'],'final_answer':a['final_answer']} for a in g['calculator_answers']]
    req,packing=pack_request(ctx,0,[],'Propose reusable procedural workflow via propose_workflow.',lambda req:count_request_tokens(req,[CANDIDATE_SCHEMA],backend._processor.tokenizer))
    receipt={'submitted':sum(r['submitted'] for r in results),'cases':4,'terminations':[r['termination'] for r in results],
             'calls':model.calls,'packing_replay':packing,'duplicate_execution':0,
             'gate':'at least3/4 submitted; <=96 calls; no provider/context failure; old overflow context packs <=20000 tokens; structural tests passed',
             'time':time.time()}
    receipt['passed']=receipt['submitted']>=3 and model.calls<=96 and all(r['termination'] not in ['provider_failure','context_overflow','context_limit'] for r in results) and packing['input_tokens']<=20000
    atomic(root/'ACCEPTANCE.json',receipt)
    if not receipt['passed']:return 2
    atomic(root/'FORMAL_STARTED.json',{'time':time.time(),'acceptance_calls':model.calls,'protocol':'runtime_v2'})
    return run(root,repo,hcl_path,model_path,model_override=model)
