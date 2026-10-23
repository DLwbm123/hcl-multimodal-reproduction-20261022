"""Task-aware retrieval adaptation; source task is never inferred from answers."""
from contextvars import ContextVar
from types import MethodType

TASKS = frozenset(('classification', 'polyp_localization', 'visual_qa'))

def eligible_ids(records, task):
    return {r['memory_id'] for r in records
            if r.get('memory_id') and r.get('internal_source', {}).get('task_name') == task}

def install_task_scoped_memory(pipeline, receipt=None):
    if getattr(pipeline, '_task_scoped_memory_installed', False):
        raise ValueError('already installed')
    if pipeline.harness_version or pipeline.global_batch_index or pipeline.checkpoint.has_state:
        raise ValueError('fresh pipeline required')
    if pipeline.memory.raw.records or pipeline.memory.abstract.records:
        raise ValueError('empty initial memory required')
    if pipeline.read_only_workers != 1 or pipeline.prediction_cache.enabled:
        raise ValueError('serial uncached inference required')
    active = ContextVar('source_task', default=None)
    for kind in ('raw', 'abstract'):
        memory = getattr(pipeline.memory, kind)
        original = memory.retrieval_candidates
        def filtered(self, _original=original, _kind=kind):
            task = active.get()
            if task not in TASKS:
                raise RuntimeError('retrieval outside known source task context')
            pool = _original()
            allowed = eligible_ids(self.records, task)
            result = [item for item in pool if item.get('memory_id') in allowed]
            if receipt:
                receipt({'task':task, 'kind':_kind, 'before':len(pool), 'after':len(result),
                         'all_candidates_source_task_matched':True})
            return result
        memory.retrieval_candidates = MethodType(filtered, memory)
    original_one = pipeline._predict_one
    def predict_one(self, stream_position, example, **kwargs):
        task = example.get('task_name')
        if task not in TASKS:
            raise ValueError('unknown task; refuse inferred task')
        token = active.set(task)
        try:
            return original_one(stream_position, example, **kwargs)
        finally:
            active.reset(token)
    pipeline._predict_one = MethodType(predict_one, pipeline)
    pipeline._task_scoped_memory_installed = True
    return pipeline
