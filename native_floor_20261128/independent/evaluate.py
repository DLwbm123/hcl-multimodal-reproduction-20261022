"""Compare frozen HCL snapshots on a reserved native-coordinate image panel."""
import os
from pathlib import Path
import random
from statistics import mean
import time

from hcl import cli
from hcl.evaluator import (answer_score, _compute_coco_caption_metrics, _detection_match_counts,
                           _ptb_tokenize, _strip_answer_wrapper)
from hcl.pipeline import HCLPipeline
from hcl.runtime_fail_closed import install_memory_selector_fail_closed
from hcl.totalbaseline.__main__ import _generate
from full import FloorPipeline, direct_key
from screen import Recorder, groups, metric, predictions_by_id, read, rows, save

MODES=('raw','stability','locked_route','floor')
HCL_TASKS={'coco_detection','coco_caption'}


def parts(task, examples, predictions):
    answers=predictions_by_id(predictions)
    if task=='coco_detection':
        counts=[_detection_match_counts(answers[e['task_id']],e['answer'],e) for e in examples]
        return [(2*matched, predicted+gold) for matched,predicted,gold in counts]
    if task=='coco_caption':
        gts={e['task_id']:[{'caption':str(c)} for c in e['metadata']['answers'] if str(c).strip()]
             for e in examples}
        predicted={k:[{'caption':_strip_answer_wrapper(str(v or ''))}] for k,v in answers.items()}
        scores,_=_compute_coco_caption_metrics(_ptb_tokenize(gts),_ptb_tokenize(predicted))
        scorers=[scores[k] for k in ('Bleu_4','METEOR','ROUGE_L','CIDEr')]
        return [(mean(min(max(s[e['task_id']],0.),1.) for s in scorers),1) for e in examples]
    return [(answer_score(answers[e['task_id']],e),1) for e in examples]


def ratio(values, indices):
    numerator=denominator=0
    for i in indices:
        a,b=values[i]
        numerator+=a
        denominator+=b
    return numerator/denominator if denominator else 0.


def bootstrap(values):
    rng=random.Random(26112894)
    reference=values['raw']
    all_ids=range(len(reference))
    absolute={m:[] for m in MODES}
    comparative={m:[] for m in MODES}
    for _ in range(2000):
        sample=rng.choices(all_ids,k=len(reference))
        raw=ratio(reference,sample)
        for mode,v in values.items():
            score=ratio(v,sample)
            absolute[mode].append(score)
            comparative[mode].append(score-raw)

    def range95(vector):
        vector.sort()
        return [vector[49],vector[1949]]

    return {m:dict(score_ci95=range95(absolute[m]),delta_ci95=range95(comparative[m])) for m in MODES}


def inference(pipeline, test, phase):
    version=pipeline._component_version_snapshot()
    memory=pipeline.memory.snapshot()
    enabled=set(getattr(pipeline,'harness_tasks',set()))
    result={task:pipeline._predict(xs,progress_label=f'{phase}:{task}') for task,xs in test.items()}
    assert pipeline.memory.snapshot()==memory and pipeline._component_version_snapshot()==version
    assert set(getattr(pipeline,'harness_tasks',set()))==enabled
    return result


def config_for(source, workspace, checkpoint):
    config=read(source/'configs/fail_closed_stability_hcl.json')
    config.update(storage_dir=str(Path(workspace).resolve()),reset_run_on_start=False,
                  checkpoint=dict(enabled=True,resume=True,path=str(checkpoint)),
                  prediction_cache=dict(enabled=False))
    return config


def main():
    begun=time.monotonic()
    source=Path(os.environ['SOURCE_CODE'])
    full=Path(os.environ['FULL_DIR'])
    assert (full/'exits').read_text().strip()=='0'
    assert len(read(full/'full_result.json')['matrix'])==4
    panel=read('PANEL.json')
    test=groups(Path('code/configs/taskstream_p.json').resolve(),'test',100)
    assert list(test)==list(panel['records'])
    for task,examples in test.items():
        assert [(e['source_task_id'],e['metadata']['image_id']) for e in examples]==[
                  (r['id'],r['image_id']) for r in panel['records'][task]]
    old_path=source/'storage/experiments/native_fail_closed_v1/stability_hcl/checkpoint.json'
    config=config_for(source,'old',old_path)
    mkw={k:v for k,v in config['model'].items() if k!='class_path'}
    mkw['log_path']='model.log'
    model=Recorder(**mkw)
    model.scope='raw'
    raw={k:[dict(task_id=e['task_id'],answer=_generate(model,e,e['task_id'])) for e in xs]
         for k,xs in test.items()}
    print('raw inference complete',flush=True)
    install_memory_selector_fail_closed(HCLPipeline)
    model.scope='stability'
    old=cli._build_pipeline(config,model=model,base_dir=Path.cwd())
    assert old.harness_version==3 and old.memory.snapshot()==read(old_path)['memory_snapshot']
    stability=inference(old,test,'stability')
    print('archived harness inference complete',flush=True)

    by_id={k:v for task in raw.values() for k,v in predictions_by_id(task).items()}
    exact_cache={direct_key(e):by_id[e['task_id']] for group in test.values() for e in group}
    model.scope='floor'
    floors=read(full.parent/'raw_floor.json')
    path=full/'out/checkpoint.json'
    cli.HCLPipeline=lambda *a,**k: FloorPipeline(*a,raw_floors=floors,raw_validation=exact_cache,**k)
    floor=cli._build_pipeline(config_for(source,'new',path),model=model,base_dir=Path.cwd())
    checkpoint=read(path)
    assert checkpoint['continual']['completed_stages']==4 and floor.memory.snapshot()==checkpoint['memory_snapshot']
    assert sorted(floor.harness_tasks)==read(full/'full_result.json')['enabled_harness_tasks']
    final=inference(floor,test,'floor')
    locked={k:(stability if k in HCL_TASKS else raw)[k] for k in test}
    modes=dict(raw=raw,stability=stability,locked_route=locked,floor=final)
    save('private_predictions.json',modes)
    scores={}
    for task,examples in test.items():
        contributions={}
        stats={}
        for mode,all_predictions in modes.items():
            predictions=all_predictions[task]
            if mode=='locked_route':
                predecessor='stability' if task in HCL_TASKS else 'raw'
                contributions[mode]=contributions[predecessor]
                stats[mode]=dict(stats[predecessor])
            else:
                result=metric(examples,predictions)
                assert task!='coco_caption' or result['caption_metric_source']=='pycocoevalcap'
                values=parts(task,examples,predictions)
                assert abs(ratio(values,range(len(examples)))-result['primary_score']) < 1e-7
                contributions[mode]=values
                stats[mode]=dict(score=result['primary_score'], n=result['total'])
        intervals=bootstrap(contributions)
        for mode,value in stats.items():
            value.update(delta_vs_raw=value['score']-stats['raw']['score'],**intervals[mode])
        scores[task]=stats
        print('scored',task,flush=True)
    average={mode:mean(task[mode]['score'] for task in scores.values()) for mode in MODES}
    from collections import Counter
    assert all(not p.requires_grad for p in model._model.parameters())
    save('RESULT.json',dict(by_task=scores,average=average,delta_vs_raw={k:v-average['raw'] for k,v in average.items()},
                enabled_harness_tasks=sorted(floor.harness_tasks),seconds=round(time.monotonic()-begun,1),
                calls=dict(Counter(row['scope'] for row in rows('calls.jsonl')))))


if __name__=='__main__':
    main()
