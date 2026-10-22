"""Read-only diagnosis helpers; no learning or candidate search."""
import copy
from types import MethodType

CONDITIONS=('direct','static_no_memory','raw_only','abstract_only','full_memory')

def memory_pool(snapshot,condition):
    if condition not in CONDITIONS:raise ValueError('unknown condition')
    return {'raw':copy.deepcopy(snapshot['raw']) if condition in ('raw_only','full_memory') else [],
            'abstract':copy.deepcopy(snapshot['abstract']) if condition in ('abstract_only','full_memory') else [],
            'anchor':[],'skills':[]}

def lock_learning(pipe):
    def forbidden(self,*args,**kwargs):raise PermissionError('development diagnosis forbids learning and test API')
    for name in ('train','continual_train','test'):
        setattr(pipe,name,MethodType(forbidden,pipe))
    return pipe

def learning_state(pipe):
    return copy.deepcopy((pipe.memory.snapshot(),pipe.component_versions,pipe.harness_version,len(pipe.optimizer.history)))

def pair_changes(reference,current):
    if set(reference)!=set(current):raise ValueError('paired sample IDs differ')
    return {'successful_to_failed':sum(reference[k]['success'] and not current[k]['success'] for k in reference),
            'failed_to_successful':sum(not reference[k]['success'] and current[k]['success'] for k in reference),
            'answer_changed':sum(reference[k]['answer']!=current[k]['answer'] for k in reference),
            'total':len(reference)}
