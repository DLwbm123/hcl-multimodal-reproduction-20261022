"""Reuse HCL memory with explicit, non-authoritative paired operation feedback."""
from hcl.memory import ExperienceMemory
from hcl.memory.raw import RawMemory

class EvidenceRawMemory(RawMemory):
    def retrieval_candidates(self):
        return [{'memory_id':x['memory_id'],'memory_kind':'raw','question':x['question'],
                 'previous_attempt':x['response'],'previous_attempt_correct':None,
                 'correct_feedback':x['feedback'],'feedback_type':'paired_operation_observation',
                 'output_form':'single_option_label',
                 'usage_rule':'Historical operation evidence only; unknown semantic outcomes stay unknown. '
                              'No case label or observation is evidence about the current patient. '
                              'Formatting or empty-observation gains do not prove tool utility.'}
                for x in self.records if not x.get('is_anchor')]

class EvidenceMemory(ExperienceMemory):
    def __init__(self, config, *, model):
        super().__init__(config,model=model)
        self.raw=EvidenceRawMemory(self.raw.path,capacity_per_task=config.raw_capacity_per_task)

    def record_training_interaction(self, task_name, example, prediction):
        super().record_training_interaction(task_name,example,prediction)
        for x in self.raw.records:
            if x['internal_source']['task_id']==example['task_id']:
                # The raw item is an operation record, not an answer/target pair.
                x['response_is_correct']=None
                x['format_compliant']=True
        self.raw._persist()
