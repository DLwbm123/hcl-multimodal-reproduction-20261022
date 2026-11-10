"""Generate unaccepted HCL candidates using source-learning traces only."""
import json
import os
from pathlib import Path
import signal
import time
import traceback
from copy import deepcopy
from hcl.memory import MemoryConfig
from hcl.memory.abstract import _extract_json, _normalize_abstraction
from hcl.optimizer import Optimizer
from hcl.router import DEFAULT_MEMORY_SELECTOR_TEMPLATE
from evidence_memory import EvidenceMemory

ROOT=Path(os.environ['R981_ROOT'])
BUDGET=json.loads((ROOT/'budget.json').read_text())
CALLS=json.loads((ROOT/'private/generation_calls.json').read_text()) if (ROOT/'private/generation_calls.json').exists() else []

def save(name,x):
    p=ROOT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,default=str)+'\n');t.replace(p)

def guard():
    if time.time()>=BUDGET['hard_deadline_epoch'] or (ROOT/'STOPPED.json').exists():raise TimeoutError('Original run budget exhausted')

class Controller:
    def __init__(self):
        import torch
        from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
        path=os.environ['LOCAL_MODEL_PATH']
        self.processor=AutoProcessor.from_pretrained(path,local_files_only=True)
        self.model=Qwen2_5_VLForConditionalGeneration.from_pretrained(path,torch_dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa',local_files_only=True).eval()
        self.model.requires_grad_(False)
        self.batch=[]

    def generate(self,prompt,*,state):
        import torch
        guard()
        safety=('Return only the requested JSON schema. You are deriving transferable medical-Agent process guidance '
          'from actual source-learning operations, not diagnosing the current or future patient. '
          'No independent clinical ground truth is available. Unknown benchmark outcomes must stay null/unknown. '
          'A completed tool and valid visual encoding do not establish relevance, truth, or sufficient detail. '
          'Raw gain involving a failed format or an empty observation does not demonstrate useful tool evidence. '
          'Retain losses and neutral cases. Do not copy patient facts, case identifiers, option letters or benchmark labels into abstractions. '
          'Do not claim a validated efficacy improvement, invent tool observations, or change tools, labels, guards or budgets. '
          'Learn conditions for selecting or rejecting operations and adopting or ignoring observations; do not default to calling every tool. '
          'Available native tools: chest_xray_expert, chest_xray_classifier, chest_xray_segmentation (anatomy only), chest_xray_report_generator. '
          'No usefulness evidence for an operation means unknown, not success; an empty set of justified skills is valid.')
        if state.get('memory_phase')=='abstract_memory':
            prompt+='\nComplete current learning batch (all pairs, no outcome selection):\n'+json.dumps(self.batch)
            schema={role:{'guidance':'required nonempty transferable operation rule grounded in the observed calls', 'applicable_when':'visible applicability condition', 'output_form':'single_option_label', 'reusable_pattern':'observed operation pattern', 'failure_mode':'observed failure or unknown, not invented clinical diagnosis'} for role in ['main_policy','failure_avoidance']}
            prompt+='\nMANDATORY RESPONSE SHAPE: '+json.dumps(schema)+'\nEvery role must contain guidance. No field may be omitted. Do not state that tool calls improve accuracy; these traces do not establish that. Do not introduce unavailable clinical history as a solution.'
        chat=[{'role':'system','content':safety},{'role':'user','content':prompt}]
        text=self.processor.apply_chat_template(chat,tokenize=False,add_generation_prompt=True)
        inputs=self.processor(text=[text],return_tensors='pt').to(self.model.device);length=inputs.input_ids.shape[-1]
        if length>20000:raise ValueError('Frozen metadata context limit exceeded')
        start=time.time();cap=min(4096,max(2048,int(state.get('max_new_tokens',2048))))
        with torch.inference_mode():out=self.model.generate(**inputs,max_new_tokens=cap,do_sample=False,max_time=180)
        raw=self.processor.batch_decode(out[:,length:],skip_special_tokens=True)[0].strip()
        CALLS.append({'state':state,'input_tokens':int(length),'output_tokens':int(out.shape[-1]-length),'seconds':time.time()-start,'prompt':prompt,'output':raw})
        save('private/generation_calls.json',CALLS)
        if state.get('memory_phase')=='abstract_memory':
            obj=_extract_json(raw)
            if not all(isinstance(obj.get(k),dict) and _normalize_abstraction(obj[k]) for k in ['main_policy','failure_avoidance']):
                raise ValueError('Invalid learned abstraction; generic fallback is disallowed')
        return raw


def main():
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('Independent supervisor stop')))
    data=json.loads((ROOT/'private/source_evidence.json').read_text())
    assert len(data)==46 and sum(x['feedback']['semantic_feedback_allowed'] for x in data)==13
    assert all(all(v is None for v in x['feedback']['benchmark_agreement'].values()) for x in data if not x['feedback']['semantic_feedback_allowed'])
    save('GENERATION_STARTED.json',{'time':time.time(),'original_start':BUDGET['started_epoch'],'source_count':46})
    fatal=None
    try:
        model=Controller();memory=EvidenceMemory(MemoryConfig(record_dir=str(ROOT/'private/memory'),raw_capacity_per_task=46,
             abstract_mode='llm',abstract_max_new_tokens=2048,abstract_prompt_max_chars=16000,skill_enabled=True,
             skill_min_source_tasks=2,skill_invalid_output_retries=0,anchor_selection_mode='off'),model=model)
        empty=deepcopy(memory.snapshot());updates=[]
        for group in ['single_image','multiple_images']:
            rows=[x for x in data if x['visible_group']==group]
            for start in range(0,len(rows),6):
                batch=rows[start:start+6];examples=[]
                for x in batch:
                    ex={'task_id':x['id'],'question':x['question'],'task_type':'medical_agent_operation',
                        'split':'train','answer':x['feedback']}
                    memory.record_training_interaction(group,ex,{'task_id':x['id'],'answer':x['observed_operations']});examples.append(ex)
                model.batch=batch
                updates.append(memory.consolidate_task(group,examples))
                save('GENERATION_PROGRESS.json',{'time':time.time(),'raw_count':len(memory.raw.records),'abstract_count':len(memory.abstract.records),'skill_count':len(memory.skills.records),'calls':len(CALLS)})
        model.batch=[]
        assert len(memory.raw.records)==46 and all(x['response_is_correct'] is None for x in memory.raw.records)
        assert not memory.anchor.records
        selector=json.loads(DEFAULT_MEMORY_SELECTOR_TEMPLATE.read_text())
        selector['sections']=[s.replace('correct_feedback is authoritative;','correct_feedback is non-authoritative historical operation evidence;') for s in selector['sections']]
        candidate={'version':1,'memory':memory.snapshot(),'memory_selector':selector,'accepted':False}
        save('private/candidate_memory.json',candidate)
        optimizer=Optimizer(record_dir=ROOT/'private/optimizer',candidate_count=1,components=['memory_selector'])
        generated=optimizer.generate_artifact_candidates('memory_selector',{'metrics':{'source_groups':46,'semantic_sources':13,'attributable_useful_evidence_gain':0}},model=model,current_artifact={'content':selector},
           generation_context={'abstract_memory':candidate['memory']['abstract'],'skills':candidate['memory']['skills'],
             'boundary':'No development/anchor/test labels. Select transferable operation guidance only when applicable to current visible evidence gap; otherwise select nothing. Do not use raw historical answers as current case facts.'})
        valid=[c for c in generated if not c['validation_errors']]
        names=['memory']
        if valid and valid[0]['artifact_content']!=selector:
            alternate=deepcopy(candidate);alternate['memory_selector']=valid[0]['artifact_content'];save('private/candidate_memory_selector.json',alternate);names.append('memory_selector')
        memory.restore(empty);assert memory.snapshot()==empty
        report={'source_groups':46,'semantic_sources':13,'unknown_sources':33,'raw_memory_count':46,'abstract_count':len(candidate['memory']['abstract']),
             'skill_count':len(candidate['memory']['skills']),'candidate_names':names,'accepted':False,'restored_empty_state':True,'generation_calls':len(CALLS),
             'input_tokens':sum(c['input_tokens'] for c in CALLS),'output_tokens':sum(c['output_tokens'] for c in CALLS),'selector_parse_status':optimizer.last_generation_diagnostics.get('parse_status'),
             'development_or_test_targets_accessed':False,'elapsed_seconds':time.time()-BUDGET['started_epoch'],'method_efficacy_established':False}
        save('GENERATION_REPORT.json',report)
    except BaseException as exc:
        fatal=repr(exc);save('FAILED.json',{'error':fatal,'traceback':traceback.format_exc()})
    save('GENERATION_FINISHED.json',{'fatal':fatal,'calls':len(CALLS),'time':time.time(),'accepted':False})
    return 2 if fatal else 0

if __name__=='__main__':raise SystemExit(main())
