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
        protocol=StubProtocol('✿FUNCTION✿: calculate\n✿ARGS✿: {"expression":"2+3","variables":{}}\n✿RESULT✿: invented')
        response=protocol.chat(to_qwen(messages),functions=[s['function'] for s in SCHEMAS],stream=False)
        decoded=self.bridge().decode(response)
        self.assertEqual(decoded.tool_calls[0]['args']['expression'],'2+3')
        self.assertNotIn('invented',str(response))
        rendered='\n'.join(m.content for m in protocol.seen)
        self.assertIn('✿RESULT✿: {"rows": []}',rendered)
        self.assertIn('✿FUNCTION✿: read_facts',rendered)
    def test_vl25_nous_protocol_and_tool_observation(self):
        protocol=StubProtocol('<tool_call>\n{"name":"calculate","arguments":{"expression":"2+3","variables":{}}}\n</tool_call>')
        from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import NousFnCallPrompt
        protocol.fncall_prompt=NousFnCallPrompt()
        protocol.generate_cfg['stop']=[]
        messages=[HumanMessage(content='Read facts'),AIMessage(content='',tool_calls=[{'name':'read_facts','args':{},'id':'x'}]),ToolMessage(content='{"rows": []}',name='read_facts',tool_call_id='x')]
        result=protocol.chat(to_qwen(messages),functions=[s['function'] for s in SCHEMAS],stream=False)
        self.assertEqual(self.bridge().decode(result).tool_calls[0]['name'],'calculate')
        self.assertIn('<tool_response>\n{"rows": []}\n</tool_response>','\n'.join(m.content for m in protocol.seen))

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
        protocol=StubProtocol('✿FUNCTION✿: calculate\n✿ARGS✿: {"expression":"2+3","variables":{}}')
        result=protocol.chat(to_qwen(messages),functions=[s['function'] for s in SCHEMAS],stream=False)
        self.assertEqual(self.bridge().decode(result).tool_calls[0]['name'],'calculate')
        self.assertIn('error: malformed tool call','\n'.join(m.content for m in protocol.seen))

if __name__=='__main__': unittest.main()
