"""Opt-in protocol: exhausted memory-selector schema errors are failed samples.

The invalid selection is never accepted or used for answering. No extra retry,
role coercion, successful memoryless fallback, or gate relaxation is performed.
"""
import json
from .progress import progress

def install_memory_selector_fail_closed(pipeline_class):
    original = pipeline_class._predict_one
    if getattr(original, '_memory_selector_fail_closed', False):
        return
    def guarded(self, stream_position, example, *args, **kwargs):
        try:
            return original(self, stream_position, example, *args, **kwargs)
        except ValueError as exc:
            if not str(exc).startswith('Memory Selector LLM output remained invalid after '):
                raise
            prediction = {
                'task_id': str(example.get('task_id', '')),
                'answer': '',
                'trace': {
                    'task_interface_mode': 'intermediate_schema_failed',
                    'selected_memory': [],
                    'intermediate_failure': {
                        'phase': 'memory_selector',
                        'error': str(exc),
                        'sample_failed': True,
                        'answering_skipped': True,
                    },
                },
            }
            if kwargs.get('update_memory', False):
                self.memory.record_training_interaction(str(example.get('task_name', '')), example, prediction)
            record = {'task_id': prediction['task_id'], 'scope': kwargs.get('progress_label'),
                      'update_memory': bool(kwargs.get('update_memory', False)), 'error': str(exc),
                      'answer': '', 'policy': 'failed_sample_no_answering_no_additional_retry'}
            with (self.evaluation_recorder.record_dir / 'intermediate_failures.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
            progress(f"sample failed task_id={prediction['task_id']} reason=memory_selector_invalid_schema answering_skipped=true")
            return prediction
    guarded._memory_selector_fail_closed = True
    pipeline_class._predict_one = guarded
