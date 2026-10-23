"""The only A/B intervention: programmatic complete raw-input acquisition."""
import json
import bootstrap_v7
from adapter_v6 import adapter_class as component_adapter


def collect_raw_input(state, arm):
    receipts = []
    if arm == 'B':
        while state.next_offset() < len(state.tools.rows):
            args = {'offset':state.next_offset(), 'limit':4}
            result, error = state.call('read_facts', args)
            if error:
                raise RuntimeError('Programmatic raw-input read failed')
            receipts.append({'tool':'read_facts','args':args,'result':result})
    elif arm != 'A':
        raise ValueError('Unknown acquisition condition')
    return receipts


def adapter_class(upstream):
    class PairedHarness(component_adapter(upstream)):
        def __init__(self,*args,case,**kwargs):
            super().__init__(*args,case=case,**kwargs)
            self.programmatic_reads = collect_raw_input(self.execution, case['arm'])
            self.seed_tool_requests = self.execution.request_count
            self.auto_read_bytes = len(json.dumps(self.programmatic_reads,ensure_ascii=False).encode()) if self.programmatic_reads else 0
            self.initial_execution_state = self.execution.snapshot()
            self.limit = 12
            self.manifest.update(protocol='input_ablation_v7', input_arm=case['arm'])
    return PairedHarness
