"""Expose precise errors through both runtime and invalid-schema paths."""
import copy
import json
import bootstrap_v10
from adapter_v6 import adapter_class as previous_adapter, SYSTEM
from adapter_v9 import GUIDANCE
from adapter_v7 import collect_raw_input
from runtime_v10 import RepairState
from langchain_core.messages import SystemMessage,AIMessage,ToolMessage

REPAIR = ('When a calculation is rejected, inspect the reported parameter, source and operation errors. '
    'If the necessary evidence exists, make at most one corrected execute_calculator call, then explicitly submit its immutable result if successful. '
    'Recheck other bindings too: locally_consistent is not a semantic correctness certificate. Do not change a supported binding without evidence, and do not assume a tool rejection means missing evidence. '
    'You may abstain when required evidence is truly absent or conflicting. After the corrected call fails, the episode terminates; do not retry indefinitely.')


def adapter_class(upstream):
    class RepairHarness(previous_adapter(upstream)):
        def __init__(self,*args,case,**kwargs):
            super().__init__(*args,case=case,**kwargs)
            self.limit=2 if case['experiment_stage']=='B' else 12
            self.execution=RepairState(self.restricted,case['arm'] in ('B','D'),self.limit)
            self.preamble[0]=SystemMessage(content=SYSTEM+'\n\n'+GUIDANCE+('\n\n'+REPAIR if case['arm'] in ('C','D') else ''))
            self.preamble_chars=sum(len(m.content) for m in self.preamble)
            self.programmatic_reads=collect_raw_input(self.execution,'B')
            self.auto_read_bytes=len(json.dumps(self.programmatic_reads,ensure_ascii=False).encode())
            self.seed_feedback=None
            if case['experiment_stage']=='B':
                for index,(name,arguments) in enumerate([('get_calculator',{'calculator_id':case['target']['calculator_id']}),('execute_calculator',case['seed_call'])]):
                    value,error=self.execution.call(name,arguments)
                    if name=='execute_calculator':assert error and not self.execution.results;self.seed_feedback=value
                    callid=f'seed-{index}';eventid=self.log.emit('constructed_seed',tool=name,args=arguments,not_model_generated=True)
                    self.history.append({'event_id':eventid,'role':'ai','message':AIMessage(content='',tool_calls=[{'name':name,'args':copy.deepcopy(arguments),'id':callid,'type':'tool_call'}])})
                    eventid=self.log.emit('constructed_seed_observation',tool=name,not_model_generated=True)
                    self.history.append({'event_id':eventid,'role':'tool','tool_name':name,'message':ToolMessage(content=json.dumps(value),tool_call_id=callid,name=name,status='error' if error else 'success')})
            self.seed_tool_requests=self.execution.request_count
            self.initial_execution_state=self.execution.snapshot()
            self.manifest.update(protocol='bounded_repair_v10',feedback='precise' if self.execution.precise else 'generic',repair_instruction=case['arm'] in ('C','D'),experiment_stage=case['experiment_stage'])

        async def node_prepare(self,state):
            if self.execution.repair_exhausted:return {'termination':'repair_exhausted'}
            return await super().node_prepare(state)

        async def node_dispatch(self,state):
            ai=self.history[-1]['message']
            calls=[{**c,'invalid':False} for c in ai.tool_calls]+[{**c,'invalid':True} for c in ai.invalid_tool_calls]
            for call in calls:
                name=call.get('name') or 'unknown';args=call.get('args');callid=call['id']
                if self.execution.repair_exhausted:
                    result,error={'error':'repair_exhausted','executed':False},True
                elif call['invalid'] and name=='execute_calculator':
                    try:parsed=json.loads(args)
                    except (ValueError,TypeError):result,error=self.execution.malformed(call['error'])
                    else:result,error=self.execution.call(name,parsed)
                elif call['invalid']:result,error={'error':'malformed tool call','detail':call['error']},True
                else:result,error=self.execution.call(name,args)
                observation=json.dumps(result,ensure_ascii=False,allow_nan=False)
                eid=self.log.emit('tool_call_failed' if error else 'tool_call_completed',tool=name,args=args,call_id=callid,chars=len(observation))
                self.history.append({'event_id':eid,'role':'tool','tool_name':name,'message':ToolMessage(content=observation,tool_call_id=callid,name=name,status='error' if error else 'success')})
            return {'termination':'repair_exhausted' if self.execution.repair_exhausted else None}
    return RepairHarness
