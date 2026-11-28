"""Run the original Stability loop with a validation-qualified direct fallback."""
from collections import Counter
import os
from pathlib import Path
import time

from hcl import cli
from hcl.pipeline import HCLPipeline, _primary_metric_score
from hcl.runtime_fail_closed import install_memory_selector_fail_closed
from hcl.totalbaseline.__main__ import _generate, _raw_visible_input
from screen import Recorder, groups, metric, predictions_by_id, read, rows, save


def direct_key(example):
    return example['task_id'], example.get('image'), _raw_visible_input(example)


class FloorPipeline(HCLPipeline):
    def __init__(self, *args, raw_floors, raw_validation, **kwargs):
        super().__init__(*args, **kwargs)
        self.raw_floors = raw_floors
        self.direct_answers = dict(raw_validation)
        self.harness_tasks = set(self.checkpoint.state.get('artifacts', {}).get('enabled_harness_tasks', []))
        self.candidate_task = None
        self.route_counts = Counter()

    def _checkpoint_artifacts(self):
        return {**super()._checkpoint_artifacts(), 'enabled_harness_tasks': sorted(self.harness_tasks)}

    def _direct(self, example, route):
        task_id = example['task_id']
        key = direct_key(example)
        if key not in self.direct_answers:
            self.direct_answers[key] = _generate(self.model, example, task_id)
        self.route_counts[route] += 1
        return dict(task_id=task_id, answer=self.direct_answers[key],
                    trace=dict(task_id=task_id, route=route, task_interface_mode='raw_visible_input',
                               selected_memory=[], selected_skills=[], selected_tools=[]))

    def _predict_one(self, stream_position, example, *args, **kwargs):
        task = example['task_name']
        if task in self.harness_tasks or task == self.candidate_task:
            return super()._predict_one(stream_position, example, *args, **kwargs)
        prediction = self._direct(example, 'raw_default')
        if kwargs.get('update_memory', False):
            self.memory.record_training_interaction(task, example, prediction)
        return prediction

    def _raw_router_option(self, example):
        return lambda: self._direct(example, 'raw_direct')

    def _predict_with_router(self, *args, **kwargs):
        previous = self.model.scope
        self.model.scope = kwargs.get('progress_label','online').split(':')[0]
        try:
            return super()._predict_with_router(*args, **kwargs)
        finally:
            self.model.scope = previous

    def _evaluate_component_candidate(self, candidate, **kwargs):
        task, = {e['task_name'] for e in kwargs['current_val_examples']}
        old = self.candidate_task
        self.candidate_task = task
        try:
            return super()._evaluate_component_candidate(candidate, **kwargs)
        finally:
            self.candidate_task = old

    def _apply_accepted_candidate(self, component, artifact_path):
        task = self._active_context_value('current_task')
        assert task in self.raw_floors
        super()._apply_accepted_candidate(component, artifact_path)
        self.harness_tasks.add(task)

    def _evaluate_gate(self, *args, **kwargs):
        current = kwargs['current_val_examples']
        task, = {e['task_name'] for e in current}
        assert len(current)==10
        result = super()._evaluate_gate(*args, **kwargs)
        result['zero_shot_floor'] = self.raw_floors[task]
        return result

    def _memory_candidate_decision(self, stable, candidate, **kwargs):
        _, reasons = super()._memory_candidate_decision(stable, candidate, **kwargs)
        if _primary_metric_score(candidate['current_task_metrics'])+1e-12 < candidate['zero_shot_floor']:
            reasons.append('zero_shot_floor_not_met')
        self.route_counts['memory_rejected_below_floor'] += int('zero_shot_floor_not_met' in reasons)
        return not reasons, reasons

    def test(self, *args, **kwargs):
        before = self.memory.snapshot(), self._component_version_snapshot(), set(self.harness_tasks)
        result = super().test(*args, **kwargs)
        assert before == (self.memory.snapshot(), self._component_version_snapshot(), self.harness_tasks)
        return result


def main():
    started=time.monotonic()
    source=Path(os.environ['SOURCE_CODE']).resolve()
    screening=Path(os.environ['SCREEN_DIR']).resolve()
    config=read(source/'configs/fail_closed_stability_hcl.json')
    stream=source/config['task_stream_path']
    config['task_stream_path']=str(stream)
    floors=read(screening/'raw_floor.json')
    raw_validation=read(screening/'raw_validation_private.json')
    validation=groups(stream,'validation',10)
    assert set(validation)==set(raw_validation)==set(floors) and len(floors)==4
    for task,v in validation.items():
        assert len(v)==10 and abs(metric(v,raw_validation[task])['primary_score']-floors[task])<1e-8
    by_id={key:value for preds in raw_validation.values() for key,value in predictions_by_id(preds).items()}
    cached={direct_key(example):by_id[example['task_id']] for v in validation.values() for example in v}
    assert len(cached)==40
    output=Path('out').resolve()
    output.mkdir(exist_ok=True)
    config.update(storage_dir=str(output),reset_run_on_start=False,
                  checkpoint=dict(enabled=True,resume=True,path=str(output/'checkpoint.json')),
                  prediction_cache=dict(enabled=True,path=str(output/'predictions.jsonl')))
    base_interface=read('hcl/task_interface/structuring_prompt_default.json')
    base_interface['template_id']='raw_visible_start'
    base_interface['sections'].insert(-2,'Copy the original question verbatim into the text evidence. '
                                         'Keep its requested answer format and visible image reference. '
                                         'Use only constraints explicitly present in the original interaction.')
    save('interface.json',base_interface)
    config['task_interface']['structuring_template_path']=str(Path('interface.json').resolve())
    values={k:v for k,v in config['model'].items() if k!='class_path'}
    values['log_path']=str(output/'model.log')
    model=Recorder(**values)
    model.scope='optimization'
    install_memory_selector_fail_closed(FloorPipeline)
    cli.HCLPipeline=lambda *a,**k: FloorPipeline(*a,raw_floors=floors,raw_validation=cached,**k)
    pipeline=cli._build_pipeline(config,model=model,base_dir=Path.cwd())
    assert pipeline.optimizer.selection_objective=="plasticity"
    assert pipeline.optimizer.historical_loss_budget==0 and pipeline.optimizer.min_primary_score_delta>0
    assert pipeline.read_only_workers==1
    for step in cli._select_flow_steps(config,{'train','test'}):
        cli._run_step(config,step=step,pipeline=pipeline,base_dir=Path.cwd(),limit_per_task=None)
    import torch
    assert all(not p.requires_grad for p in model._model.parameters())
    checkpoint=pipeline.checkpoint.state['continual']
    assert checkpoint['completed_stages']==4
    assert all(len(v)==100 for v in checkpoint['previous_predictions'].values())
    gates=read(output/'evaluator/gate_validation_history.json')
    accepted=[]
    reasons=Counter()
    for gate in gates:
        task=gate['current_task']
        for candidate in gate['candidates']:
            reasons.update(candidate['decision_reasons'])
            if candidate['decision']!='accepted':
                continue
            original=float(gate['baseline']['current_task_metrics']['primary_score'])
            score=float(candidate['current_task_metrics']['primary_score'])
            assert score>original+1e-12 and score+1e-12>=floors[task]
            assert candidate['historical_anchor_metrics']['forget_count']==0
            accepted.append(dict(task=task,component=gate['component'],previous=original,score=score,
                                 floor=floors[task]))
    order=list(floors)
    matrix=checkpoint['primary_score_rows']
    assert len(matrix)==4 and len(matrix[-1])==4
    final=matrix[-1]
    forgetting=sum(max(row[i] for row in matrix if row[i] is not None)-final[i] for i in range(3))/3
    save('full_result.json',dict(tasks=order, matrix=matrix, average=sum(final)/4, average_forgetting=forgetting,
             validation_floors=floors, gate_rounds=len(gates), accepted=accepted, rejected_reason_counts=dict(reasons),
             enabled_harness_tasks=sorted(pipeline.harness_tasks),routes=dict(pipeline.route_counts),
             model_calls=dict(Counter(x['scope'] for x in rows('calls.jsonl'))),
             peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,seconds=time.monotonic()-started))


if __name__=='__main__':
    main()
