import os,sys,time,json,copy,traceback
from pathlib import Path
ROOT=Path(os.environ['HCL_DIAGNOSTIC_ROOT']).resolve()
PREPARED=Path(os.environ['HCL_PREPARED_ROOT']).resolve()
REPO=Path(__file__).resolve().parents[2]
P=json.loads((ROOT/'protocol.json').read_text());B=json.loads((ROOT/'budget.json').read_text())
os.environ.update(HF_HOME=str(ROOT/'cache'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TMPDIR=str(ROOT/'tmp'),TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',CUDA_CACHE_PATH=str(ROOT/'cache/cuda'),XDG_CACHE_HOME=str(ROOT/'cache'),TORCH_HOME=str(ROOT/'cache/torch'),TRITON_CACHE_DIR=str(ROOT/'cache/triton'))
for d in ['cache','tmp']:(ROOT/d).mkdir(exist_ok=True)
import tempfile
tempfile.tempdir=str(ROOT/'tmp')
sys.path[:0]=[str(Path(__file__).parent),str(REPO/'medical_transfer/coordinate_interface'),str(REPO/'medical_transfer/src'),str(REPO)]
from integration import CoordinateBoundary,install_coordinate_boundary
from diagnostic_support import memory_pool,lock_learning,learning_state,pair_changes,CONDITIONS
from runtime_support import install_protocol
from hcl.models import LocalTransformersModel,_state_phase
from medical_hcl.bridge import construct_pipeline
from medical_hcl.evaluation import MedicalEvaluator,score_case
import torch

def write(name,obj):
    f=ROOT/name;temp=f.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2,allow_nan=False));temp.replace(f)

def append(name,obj):
    with (ROOT/name).open('a') as f:f.write(json.dumps(obj,allow_nan=False)+'\n')

class MeteredBoundary(CoordinateBoundary):
    scope='loading';condition='none';calls=0
    def generate(self,prompt,*,state):
        if time.time()-B['started_epoch']>=B['early_stop_seconds']:raise RuntimeError('diagnostic deadline')
        if self.calls>=B['max_calls']:raise RuntimeError('diagnostic call cap')
        self.calls+=1;write('model_attempts.json',{'attempts':self.calls,'started_epoch':B['started_epoch']})
        started=time.monotonic();ok=False
        try:
            answer=super().generate(prompt,state=state);ok=True;return answer
        finally:
            append('call_costs.jsonl',{'attempt':self.calls,'condition':self.condition,'phase':_state_phase(state),'seconds':time.monotonic()-started,'success':ok,'peak_allocated_bytes':torch.cuda.max_memory_allocated()})

try:
    backend=LocalTransformersModel(name='Qwen2.5-VL-7B-Instruct',path=B['model_path'],torch_dtype='bfloat16',attn_implementation='sdpa',max_new_tokens=2048,answer_max_new_tokens=512,temperature=0.,enable_thinking=False,local_files_only=True)
    backend._load();assert not backend._model.training and all(not p.requires_grad for p in backend._model.parameters())
    model=MeteredBoundary(backend,ROOT/'private_coordinate_trace.jsonl')
    if (ROOT/'model_attempts.json').exists():
        model.calls=json.loads((ROOT/'model_attempts.json').read_text())['attempts']
    rows=[json.loads(x) for x in (ROOT/'private_development.jsonl').read_text().splitlines()]
    snapshot=json.loads((ROOT/'private_frozen_memory.json').read_text())
    vocabulary=json.loads((ROOT/'class_vocabulary.json').read_text());evaluator=MedicalEvaluator(vocabulary)
    assert len(rows)==36 and all(e['metadata']['role']=='validation' for e in rows)
    saved=[json.loads(x) for x in (ROOT/'private_predictions.jsonl').read_text().splitlines()] if (ROOT/'private_predictions.jsonl').exists() else []
    if saved:
        assert len(saved)==36 and {x['condition'] for x in saved}=={'direct'}
        assert {x['prediction']['task_id'] for x in saved}=={e['task_id'] for e in rows}
        assert model.calls==36
    all_results={};paired_cases={}
    for condition in CONDITIONS:
        model.condition=condition;predictions=[]
        pipe=None
        if condition!='direct':
            pipe=construct_pipeline(PREPARED,ROOT/'state'/condition,model=model,real_execution_authorized=True,arm='stability_hcl')
            install_protocol(pipe,model,ROOT,{'anchors_per_task':24})
            install_coordinate_boundary(pipe,model)
            lock_learning(pipe)
            # Same config/H in all four harness conditions; only the candidate pool differs.
            pipe.memory.restore(memory_pool(snapshot,condition))
            before=learning_state(pipe)
            config={'router':pipe.router_config.to_dict(),'interface':pipe.task_interface_config.to_dict()}
            config['router']={k:v for k,v in config['router'].items() if 'path' not in k}
            config['interface']={k:v for k,v in config['interface'].items() if 'path' not in k}
            if condition=='static_no_memory':frozen_config=copy.deepcopy(config)
            else:assert frozen_config==config
        write('progress.json',{'condition':condition,'completed':0,'total':36,'calls':model.calls})
        for index,e in enumerate(rows):
            if condition=='direct' and saved:
                prediction=next(x['prediction'] for x in saved if x['prediction']['task_id']==e['task_id'])
                predictions.append(prediction)
                continue
            if condition=='direct':
                visible=model.prepare_example(e)
                state={'question':visible['question'],'internal_sample_id':e['task_id'],'task_context':{'files':[{'type':'image','path':e['image']}]}}
                answer=model.generate(visible['question'],state=state)
                prediction={'task_id':e['task_id'],'answer':answer,'trace':{'selected_memory':[],'prompt':visible['question']}}
            else:
                prediction=pipe._predict([e],update_memory=False,progress_label='development:'+condition)[0]
                assert learning_state(pipe)==before
            predictions.append(prediction)
            valid,success,score,aux=score_case(e,prediction['answer'],vocabulary)
            trace=prediction.get('trace',{})
            append('private_predictions.jsonl',{'condition':condition,'task':e['task_name'],'prediction':prediction,'format_valid':valid,'success':success,'score':score,'auxiliary':aux})
            append('read_only_checks.jsonl',{'condition':condition,'sample_index':index,'learning_state_unchanged':True,'parameters_frozen':all(not p.requires_grad for p in backend._model.parameters())})
            write('progress.json',{'condition':condition,'completed':index+1,'total':36,'calls':model.calls})
        metrics={};cases={}
        for task in P['tasks']:
            subset=[e for e in rows if e['task_name']==task]
            ids={e['task_id'] for e in subset}
            evaluated=evaluator.evaluate(subset,[p for p in predictions if p['task_id'] in ids],split='development:'+condition+':'+task)
            assert evaluated['metrics']['total']==12
            metrics[task]=evaluated['metrics']
            by={x['task_id']:x for x in evaluated['case_results']}
            pp={x['task_id']:x['answer'] for x in predictions}
            cases[task]={k:{'success':v['success'],'answer':pp[k]} for k,v in by.items()}
        all_results[condition]={'metrics':metrics,'mean_success_percent':sum(m['primary_score'] for m in metrics.values())/3*100}
        paired_cases[condition]=cases
        write('partial_results.json',all_results)
    changes={}
    for condition in CONDITIONS:
        if condition=='static_no_memory':continue
        changes[condition]={task:pair_changes(paired_cases['static_no_memory'][task],paired_cases[condition][task]) for task in P['tasks']}
    write('RESULT.json',{'conditions':all_results,'paired_changes_vs_static_no_memory':changes,'static_and_no_memory_same_condition_reused':True,'test_used':False,'training_runs':0,'model_calls':model.calls,'parameters_frozen':True,'memory_source':'prior committed VQA memory; fixed before new development outputs','coordinate_semantic_acceptance_previously_failed':True})
    write('EXIT.json',{'exit_code':0,'ended':time.time(),'elapsed_seconds':time.time()-B['started_epoch'],'calls':model.calls,'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
except BaseException:
    (ROOT/'FAILURE.txt').write_text(traceback.format_exc());write('EXIT.json',{'exit_code':1,'ended':time.time()});raise
