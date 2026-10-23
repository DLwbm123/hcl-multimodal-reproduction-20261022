"""Identical raw acquisition with one explicit operand-schema intervention."""
import json
import bootstrap_v9
from adapter_v6 import adapter_class as previous_adapter
from adapter_v7 import collect_raw_input
from runtime_v9 import OperandState
from langchain_core.messages import SystemMessage

GUIDANCE = ('For execute_calculator, use its explicit operand schema: give parameter, value, operation and operands. '
            'Every operand must contain fact_id and its source value. Choose identity, lowercase or multiply as appropriate. '
            'Include every quantity used to derive the parameter. Do not use the older fact_ids/transformation format; the ledger generates these from your declared operands. '
            'The tool checks the declared chain only; you remain responsible for selecting correct fields and all required operands.')


def adapter_class(upstream):
    class BindingHarness(previous_adapter(upstream)):
        def __init__(self,*args,case,**kwargs):
            if case['arm'] not in ('R','I'):raise ValueError('Unknown arm')
            super().__init__(*args,case={**case,'arm':'B'},**kwargs)
            if case['arm']=='I':
                self.execution=OperandState(self.restricted)
                self.preamble[0]=SystemMessage(content=self.preamble[0].content+'\n\n'+GUIDANCE)
            self.programmatic_reads=collect_raw_input(self.execution,'B')
            self.seed_tool_requests=self.execution.request_count
            self.auto_read_bytes=len(json.dumps(self.programmatic_reads,ensure_ascii=False).encode())
            self.initial_execution_state=self.execution.snapshot()
            self.limit=12
            self.preamble_chars=sum(len(m.content) for m in self.preamble)
            self.manifest.update(protocol='operand_binding_v9',input_arm='B',binding_arm=case['arm'])
    return BindingHarness
