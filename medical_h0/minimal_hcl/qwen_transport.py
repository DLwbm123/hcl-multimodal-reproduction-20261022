"""Qwen-Agent's official function-call protocol over local Transformers generation.

Only the inference backend and LangChain message bridge are local code. Protocol
formatting, assistant continuation and response parsing remain in Qwen-Agent.
This adapter accepts text-only MedTool inputs; it does not claim visual acceptance.
"""
import json
import time
from pathlib import Path

from qwen_agent.llm.function_calling import BaseFnCallModel
from qwen_agent.llm.schema import Message
from jsonschema import Draft202012Validator, ValidationError


class RunStopped(BaseException):
    pass


class TransformersFnModel(BaseFnCallModel):
    def __init__(self, backend, root):
        super().__init__({'model': 'Qwen2.5-VL-7B-Instruct', 'generate_cfg': {
            'fncall_prompt_type': 'nous', 'parallel_function_calls': False,
            'max_retries': 0, 'max_input_tokens': 0, 'seed': 42}})
        self.backend, self.root, self.calls = backend, Path(root), 0

    def _chat_stream(self, *args, **kwargs):
        raise NotImplementedError('Non-streaming local acceptance only')

    def _chat_no_stream(self, messages, generate_cfg):
        import torch
        from transformers import StopStringCriteria, StoppingCriteriaList
        budget = json.loads((self.root/'budget.json').read_text())
        guard = json.loads((self.root/'guard_status.json').read_text())
        if time.time()-guard['time'] > 30 or (self.root/'HARD_STOP.json').exists():
            raise RunStopped('guard unavailable')
        if time.time() >= budget['early_stop_epoch'] or self.calls+budget.get('previous_model_calls',0) >= budget['max_model_calls']:
            raise RunStopped('model budget reached')
        self.calls += 1
        tmp = self.root/'attempts.tmp'
        tmp.write_text(json.dumps({'calls': self.calls+budget.get('previous_model_calls',0)})); tmp.replace(self.root/'attempts.json')
        start = time.time()
        self.backend._load()
        tokenizer = self.backend._processor.tokenizer
        chat = [{'role': m.role, 'content': m.content} for m in messages]
        if any(type(m['content']) is not str for m in chat):
            raise TypeError('This MedTool bridge requires text content')
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(prompt, return_tensors='pt').to(self.backend._model.device)
        n = encoded['input_ids'].shape[-1]
        if n+1024 > 16384:
            raise ValueError('context length budget; no silent truncation')
        stop = generate_cfg.get('stop', [])
        criteria = StoppingCriteriaList([StopStringCriteria(tokenizer, stop)]) if stop else None
        with torch.inference_mode():
            generated = self.backend._model.generate(**encoded, max_new_tokens=1024,
                do_sample=False, stopping_criteria=criteria)
        output = tokenizer.decode(generated[0,n:], skip_special_tokens=True)
        with (self.root/'model_calls.jsonl').open('a') as f:
            f.write(json.dumps({'call': self.calls, 'seconds': time.time()-start,
                'input_tokens': n, 'output_tokens': generated.shape[-1]-n,
                'output': output, 'stop_words': stop})+'\n')
        self.last_usage={'input_tokens':n,'output_tokens':generated.shape[-1]-n,'total_tokens':generated.shape[-1]}
        self.last_finish='length' if generated.shape[-1]-n>=1024 else 'stop'
        # BaseFnCallModel handles stop-word removal and official function parsing.
        return [Message(role='assistant', content=output)]


def to_qwen(messages):
    """Preserve named tool results and validate correlation before conversion."""
    converted, pending = [], {}
    for m in messages:
        role = {'human': 'user', 'ai': 'assistant'}.get(m.type, m.type)
        if type(m.content) is not str:
            raise TypeError('Text-only MedTool messages required')
        if role == 'tool':
            name = pending.pop(m.tool_call_id, None)
            if name is None or name != m.name:
                raise ValueError('Mismatched tool result identity')
            converted.append({'role': 'function', 'name': name, 'content': m.content})
            continue
        if pending:
            raise ValueError('Missing tool observation')
        if m.content:
            converted.append({'role': role, 'content': m.content})
        for call in getattr(m, 'tool_calls', []) or []:
            if call['id'] in pending: raise ValueError('Duplicate tool call ID')
            pending[call['id']] = call['name']
            converted.append({'role': 'assistant', 'content': '', 'function_call': {
                'name': call['name'], 'arguments': json.dumps(call['args'], allow_nan=False)}})
        for call in getattr(m, 'invalid_tool_calls', []) or []:
            pending[call['id']] = call['name']
            converted.append({'role': 'assistant', 'content': '', 'function_call': {
                'name': call['name'], 'arguments': call['args']}})
    if pending: raise ValueError('Missing final tool observation')
    return converted


class LocalModel:
    def __init__(self, backend, root):
        self.protocol = TransformersFnModel(backend, root)
        self.schemas = []

    @property
    def calls(self): return self.protocol.calls

    def bind_tools(self, schemas):
        self.schemas = schemas
        self.validators = {s['function']['name']: Draft202012Validator(s['function']['parameters']) for s in schemas}
        return self

    def decode(self, responses):
        from langchain_core.messages import AIMessage
        content, valid, invalid = [], [], []
        for i, response in enumerate(responses):
            if response.get('content'): content.append(response['content'])
            call = response.get('function_call')
            if not call: continue
            name, arguments = call['name'], call['arguments']
            call_id = f'{self.calls}-{i}'
            try:
                args = json.loads(arguments)
                if type(args) is not dict: raise ValueError('Arguments must be an object')
                if name not in self.validators: raise ValueError('Unregistered tool')
                self.validators[name].validate(args)
                json.dumps(args, allow_nan=False)
            except (ValueError, KeyError, TypeError, ValidationError) as exc:
                invalid.append({'name': name, 'args': arguments, 'id': call_id, 'error': str(exc)[:200]})
                continue
            valid.append({'name': name, 'args': args, 'id': call_id, 'type': 'tool_call'})
        return AIMessage(content='\n'.join(content), tool_calls=valid, invalid_tool_calls=invalid)

    async def ainvoke(self, messages):
        self.last_failure=None
        try:
            responses = self.protocol.chat(messages=to_qwen(messages),
                functions=[s['function'] for s in self.schemas], stream=False)
            result = self.decode(responses)
        except (ValueError,KeyError,TypeError) as exc:
            self.last_failure='context_limit' if 'context length' in str(exc) else 'protocol_error'
            raise
        result.usage_metadata=getattr(self.protocol,'last_usage',None)
        result.response_metadata={'finish_reason':getattr(self.protocol,'last_finish','stop')}
        with (self.protocol.root/'protocol_calls.jsonl').open('a') as f:
            f.write(json.dumps({'call': self.calls, 'valid': len(result.tool_calls),
                'invalid': len(result.invalid_tool_calls), 'names': [c['name'] for c in result.tool_calls]})+'\n')
        return result
