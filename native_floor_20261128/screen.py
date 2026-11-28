"""Inspect archived outputs and select a per-task policy on the fixed validation split."""
from collections import Counter
import json
import os
from pathlib import Path
import re
from statistics import mean

from hcl import cli
from hcl.dataset import DatasetLoader, load_task_stream_specs
from hcl.evaluator import ExactMatchEvaluator, _extract_bbox, _parse_jsonish, answer_score
from hcl.native_geometry import NativePixelModel
from hcl.runtime_fail_closed import install_memory_selector_fail_closed
from hcl.pipeline import HCLPipeline
from hcl.totalbaseline.__main__ import _generate


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def save(name, data):
    p=Path(name)
    stage=p.with_suffix('.tmp')
    stage.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    stage.replace(p)


class Recorder(NativePixelModel):
    scope='raw'
    def generate(self,prompt,*,state):
        import time
        from hcl.models import _state_phase
        start=time.monotonic()
        try:
            return super().generate(prompt,state=state)
        finally:
            with open('calls.jsonl','a') as out:
                out.write(json.dumps(dict(scope=self.scope,phase=_state_phase(state),
                                          seconds=time.monotonic()-start))+'\n')


def groups(stream, split, limit):
    loader=DatasetLoader()
    return {spec['task_name']:loader.load_task_spec(spec,split=split,dataset_index=i,limit=limit)
            for i,spec in enumerate(load_task_stream_specs(stream))}


def metric(examples,predictions):
    ids={ex['task_id'] for ex in examples}; seen=[p['task_id'] for p in predictions]
    assert ids==set(seen) and len(ids)==len(seen)
    m=ExactMatchEvaluator().evaluate(examples,predictions)['metrics']
    assert m['total']==len(examples)
    return m


def subset(examples,by_id):
    return [dict(task_id=e['task_id'],answer=by_id[e['task_id']]) for e in examples]


def predictions_by_id(seq):
    out={v['task_id']:v['answer'] for v in seq}
    assert len(out)==len(seq)
    return out


def static_outputs(path):
    by_key={v['cache_key'][:12]:v['output'] for v in rows(path/'prediction_cache/read_only_predictions.jsonl')}
    assert len(by_key)==400
    log=(path/'runner.log').read_text()
    links=dict(re.findall(r'prediction cache store phase=test task_id=(\S+) key=([0-9a-f]{12})',log))
    assert len(links)==400 and set(links.values())==set(by_key)
    return {k:by_key[v] for k,v in links.items()}


def output_diagnostic(examples,by_id):
    summary=Counter(n=len(examples))
    before=[]
    after=[]
    sizes=[]
    for example in examples:
        answer=str(by_id[example['task_id']] or '')
        parsed=_parse_jsonish(answer)
        sizes.append(len(answer))
        summary.update(fenced=int(answer.lstrip().startswith('```')), multiline=int('\n' in answer.strip()),
                       over_5_words=int(len(answer.split())>5), json_object=int(isinstance(parsed,dict)),
                       json_list=int(isinstance(parsed,list)))
        reformatted=answer
        if example['task_name']=='refcoco_grounding':
            box=_extract_bbox(answer)
            summary.update(parseable_bbox=int(box is not None),
                           parseable_below_iou50=int(box is not None and answer_score(answer,example)==0))
            if box is None and isinstance(parsed,list) and len(parsed)==1 and isinstance(parsed[0],dict):
                reformatted=parsed[0]
        elif isinstance(parsed,dict) and set(parsed)=={'answer'} and isinstance(parsed['answer'],(str,int,float)):
            reformatted=parsed['answer']
        summary['mechanically_changed']+=int(reformatted!=answer)
        before.append(answer_score(answer,example)); after.append(answer_score(reformatted,example))
    return dict(summary,mean_chars=mean(sizes),primary=mean(before),after_mechanical_cleanup=mean(after))


def main():
    old=Path(os.environ['SOURCE_CODE']).resolve()
    native=old/'storage/experiments/native_pixels_50_10_100'
    raw_path=native/'raw_zero_shot'
    static_path=native/'static_harness'
    stability=old/'storage/experiments/native_fail_closed_v1/stability_hcl'
    for path in (raw_path,static_path,stability):
        assert read(path/'EXIT.json')['exit_code']==0
    config=read(old/'configs/fail_closed_stability_hcl.json')
    stream=old/config['task_stream_path']
    validation=groups(stream,'validation',10); test=groups(stream,'test',100)
    assert len(validation)==len(test)==4 and all(len(v)==10 for v in validation.values())
    assert all(len(v)==100 for v in test.values())

    raw_test=predictions_by_id(rows(raw_path/'predictions.jsonl'))
    static_test=static_outputs(static_path)
    diag={task:{arm:output_diagnostic(test[task],p) for arm,p in (('raw',raw_test),('static',static_test))}
          for task in ('refcoco_grounding','vqav2')}
    assert abs(diag['refcoco_grounding']['static']['primary']-.84)<1e-8
    assert abs(diag['vqav2']['static']['primary']-.3766666667)<1e-8
    save('output_audit.json',diag)
    print('output audit done',flush=True)

    values={k:v for k,v in config['model'].items() if k!='class_path'}
    values['log_path']='model.log'
    model=Recorder(**values)
    raw_validation={k:[dict(task_id=e['task_id'],answer=_generate(model,e,e['task_id'])) for e in group]
                    for k,group in validation.items()}
    raw_metrics={k:metric(v,raw_validation[k]) for k,v in validation.items()}
    save('raw_validation_private.json',raw_validation)
    save('raw_floor.json',{k:v['primary_score'] for k,v in raw_metrics.items()})
    print('raw validation 40 done',flush=True)

    # The old update log is scoped to contemporaneous artifacts and memory; evaluate the final checkpoint on all tasks.
    install_memory_selector_fail_closed(HCLPipeline)
    model.scope='final_validation'
    config['storage_dir']=str(Path('replay').resolve())
    config['checkpoint']=dict(enabled=True,resume=True,path=str(stability/'checkpoint.json'))
    config['prediction_cache']=dict(enabled=False)
    pipeline=cli._build_pipeline(config,model=model,base_dir=Path.cwd())
    checkpoint=read(stability/'checkpoint.json')
    before=pipeline.memory.snapshot(); versions=pipeline._component_version_snapshot()
    assert before==checkpoint['memory_snapshot'] and pipeline.harness_version==3
    final_validation={k:pipeline._predict(group,progress_label=f'final_validation:{k}') for k,group in validation.items()}
    assert pipeline.memory.snapshot()==before and pipeline._component_version_snapshot()==versions
    final_metrics={k:metric(v,final_validation[k]) for k,v in validation.items()}
    save('final_validation_private.json',final_validation)
    choice={k:'hcl' if final_metrics[k]['primary_score']>raw_metrics[k]['primary_score']+1e-12 else 'raw'
            for k in validation}
    save('choices.json',choice)
    print('validation choices locked',flush=True)

    raw_saved=read(raw_path/'metrics.json')['per_task_metrics']
    hcl_matrix=checkpoint['continual']['primary_score_rows']
    final_test={}; raw_scores={}
    for i,(k,examples) in enumerate(test.items()):
        archived=checkpoint['test'][f'test:after_vqav2:{k}']['result']['predictions']
        z=metric(examples,subset(examples,raw_test))['primary_score']
        h=metric(examples,archived)['primary_score']
        assert abs(z-raw_saved[k]['primary_score'])<1e-7 and abs(h-hcl_matrix[-1][i])<1e-7
        raw_scores[k]=z
        final_test[k]=dict(raw=z,hcl=h,selected=h if choice[k]=='hcl' else z,n=len(examples))
    matrix=[[None if s is None else (s if choice[k]=='hcl' else raw_scores[k])
             for k,s in zip(test,hist,strict=True)] for hist in hcl_matrix]
    last=matrix[-1]; forget=mean(max(row[i] for row in matrix if row[i] is not None)-last[i] for i in range(3))
    import torch
    assert all(not p.requires_grad for p in model._model.parameters())
    save('result.json',dict(diagnostics=diag,validation={k:dict(raw=raw_metrics[k]['primary_score'],
              hcl=final_metrics[k]['primary_score'],choice=choice[k],n=len(validation[k])) for k in validation},
          test=final_test,matrix=matrix,average=mean(last),average_forgetting=forget,
          calls=dict(Counter(x['scope'] for x in rows('calls.jsonl'))),
          max_allocated_mib=torch.cuda.max_memory_allocated()/2**20))


if __name__=='__main__':
    main()
