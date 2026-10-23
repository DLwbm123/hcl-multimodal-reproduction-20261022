import unittest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from qwen_agent.llm.function_calling import BaseFnCallModel
from qwen_agent.llm.schema import Message
from qwen_transport import LocalModel, to_qwen
from tools import SCHEMAS


class StubProtocol(BaseFnCallModel):
    def __init__(self, output):
        super().__init__({'generate_cfg': {'fncall_prompt_type': 'qwen', 'max_input_tokens': 0}})
        self.output, self.seen = output, None
    def _chat_stream(self,*a,**k): raise NotImplementedError
    def _chat_no_stream(self,messages,generate_cfg):
        self.seen=messages
        return [Message(role='assistant',content=self.output)]


class ProtocolTests(unittest.TestCase):
    def bridge(self): return LocalModel(None,'.').bind_tools(SCHEMAS)
    def test_official_round_trip(self):
        messages=[SystemMessage(content='Test'),HumanMessage(content='Read facts'),
            AIMessage(content='',tool_calls=[{'name':'read_facts','args':{},'id':'x'}]),
            ToolMessage(content='{"rows": []}',name='read_facts',tool_call_id='x')]
        protocol=StubProtocol('✿FUNCTION✿: calculate\n✿ARGS✿: {"episode_id":"e","expression":"2+3","variables":{}}\n✿RESULT✿: invented')
        response=protocol.chat(to_qwen(messages),functions=[s['function'] for s in SCHEMAS],stream=False)
        decoded=self.bridge().decode(response)
        self.assertEqual(decoded.tool_calls[0]['args']['expression'],'2+3')
        self.assertNotIn('invented',str(response))
        rendered='\n'.join(m.content for m in protocol.seen)
        self.assertIn('✿RESULT✿: {"rows": []}',rendered)
        self.assertIn('✿FUNCTION✿: read_facts',rendered)
    def test_vl25_nous_protocol_and_tool_observation(self):
        protocol=StubProtocol('<tool_call>\n{"name":"calculate","arguments":{"episode_id":"e","expression":"2+3","variables":{}}}\n</tool_call>')
        from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import NousFnCallPrompt
        protocol.fncall_prompt=NousFnCallPrompt()
        protocol.generate_cfg['stop']=[]
        messages=[HumanMessage(content='Read facts'),AIMessage(content='',tool_calls=[{'name':'read_facts','args':{},'id':'x'}]),ToolMessage(content='{"rows": []}',name='read_facts',tool_call_id='x')]
        result=protocol.chat(to_qwen(messages),functions=[s['function'] for s in SCHEMAS],stream=False)
        self.assertEqual(self.bridge().decode(result).tool_calls[0]['name'],'calculate')
        self.assertIn('<tool_response>\n{"rows": []}\n</tool_response>','\n'.join(m.content for m in protocol.seen))

    def test_malformed_history_not_coerced_to_string_arguments(self):
        raw='<tool_call>\n{"name":"submit_answer","arguments":{"calculators":[]}bad tail'
        invalid=AIMessage(content='',invalid_tool_calls=[{'name':'submit_answer','args':'{"calculators":[]}bad tail','id':'bad','error':'Extra data'}],response_metadata={'raw_assistant_output':raw})
        messages=[HumanMessage(content='task'),invalid,ToolMessage(content='Extra data',name='submit_answer',tool_call_id='bad',status='error')]
        converted=to_qwen(messages)
        self.assertEqual(converted[1]['content'],raw)
        self.assertNotIn('function_call',converted[1])
        self.assertEqual(converted[2]['role'],'user')
        protocol=StubProtocol('')
        from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import NousFnCallPrompt
        protocol.fncall_prompt=NousFnCallPrompt();protocol.generate_cfg['stop']=[]
        protocol.chat(converted,functions=[s['function'] for s in SCHEMAS],stream=False)
        rendered='\n'.join(m.content for m in protocol.seen)
        self.assertIn(raw,rendered)
        self.assertNotIn('"arguments": "{',rendered)

    def test_nested_source_contract_matches_execution(self):
        import json
        from test_interface import args, state
        a=args()
        def decode(a):
            return self.bridge().decode([{'function_call':{'name':'execute_calculator','arguments':json.dumps(a)}}])
        valid=decode(a)
        self.assertEqual(len(valid.tool_calls),1)
        self.assertFalse(state().call('execute_calculator',valid.tool_calls[0]['args'])[1])
        a['input_sources']['heart_rate']={'fact_id':'fact_0001','value':90}
        rejected=decode(a)
        self.assertFalse(rejected.tool_calls)
        self.assertIn('input_sources.heart_rate',rejected.invalid_tool_calls[0]['error'])
        self.assertIn('fact_ids',rejected.invalid_tool_calls[0]['error'])
        self.assertIn('transformation',rejected.invalid_tool_calls[0]['error'])
        self.assertEqual(json.loads(rejected.invalid_tool_calls[0]['args']),a)

    def test_sources_need_nonempty_reference_and_reject_extra_fields(self):
        import json
        from test_interface import args
        for source in [{'fact_ids':[],'transformation':'copy'},
                       {'fact_ids':['fact_0001'],'transformation':'copy','unit':'bpm'},
                       {'fact_ids':['fact_0001'],'transformation':''}]:
            a=args();a['input_sources']['heart_rate']=source
            result=self.bridge().decode([{'function_call':{'name':'execute_calculator','arguments':json.dumps(a)}}])
            self.assertFalse(result.tool_calls)
        a=args();a['input_sources']['heart_rate']['result_ids']=[]
        result=self.bridge().decode([{'function_call':{'name':'execute_calculator','arguments':json.dumps(a)}}])
        self.assertEqual(len(result.tool_calls),1)

    def test_missing_parameter_error_is_actionable(self):
        result=self.bridge().decode([{'function_call':{'name':'execute_calculator','arguments':'{"calculator_id":"x","episode_id":"e","patient_id":"p","input_sources":{}}'}}])
        self.assertIn("'parameters' is a required property",result.invalid_tool_calls[0]['error'])
        self.assertNotIn('Failed validating',result.invalid_tool_calls[0]['error'])

    def test_invalid_calls_fail_closed_without_repair(self):
        for name,args in [('calculate','{"expression":'),('read_file','{"path":"/etc/passwd"}'),
                          ('calculate','{"expression":"2+3","variables":{"x":true}}'),
                          ('read_facts','{"path":"secret"}')]:
            with self.subTest(name=name,args=args):
                result=self.bridge().decode([{'role':'assistant','content':'','function_call':{'name':name,'arguments':args}}])
                self.assertFalse(result.tool_calls)
                self.assertEqual(result.invalid_tool_calls[0]['args'],args)
    def test_result_identity_required(self):
        messages=[HumanMessage(content='Read'),AIMessage(content='',tool_calls=[{'name':'read_facts','args':{},'id':'x'}])]
        with self.assertRaises(ValueError): to_qwen(messages)
        with self.assertRaises(ValueError): to_qwen(messages+[ToolMessage(content='{}',name='calculate',tool_call_id='x')])
    def test_error_observation_continues_official_protocol(self):
        invalid=self.bridge().decode([{'function_call':{'name':'calculate','arguments':'{"expression":'}}])
        messages=[HumanMessage(content='Compute'),invalid,ToolMessage(content='error: malformed tool call',
            name='calculate',tool_call_id=invalid.invalid_tool_calls[0]['id'],status='error')]
        protocol=StubProtocol('✿FUNCTION✿: calculate\n✿ARGS✿: {"episode_id":"e","expression":"2+3","variables":{}}')
        result=protocol.chat(to_qwen(messages),functions=[s['function'] for s in SCHEMAS],stream=False)
        self.assertEqual(self.bridge().decode(result).tool_calls[0]['name'],'calculate')
        self.assertIn('error: malformed tool call','\n'.join(m.content for m in protocol.seen))

if __name__=='__main__': unittest.main()
