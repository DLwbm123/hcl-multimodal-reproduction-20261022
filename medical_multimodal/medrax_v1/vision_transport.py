"""Local Qwen vision transport for the unchanged MedRAX LangGraph loop."""
import json
from pathlib import Path
import re
import time

from langchain_core.messages import AIMessage


def parse_response(raw, call_prefix):
    blocks = re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>', raw, re.S)
    if raw.count('<tool_call>') != len(blocks) or raw.count('</tool_call>') != len(blocks):
        raise ValueError('Incomplete tool-call serialization')
    calls = []
    for i, block in enumerate(blocks):
        value = json.loads(block)
        if not isinstance(value, dict) or not isinstance(value.get('name'), str):
            raise ValueError('Tool call has no name')
        if not isinstance(value.get('arguments'), dict):
            raise ValueError('Tool arguments must be an object')
        calls.append({'name': value['name'], 'args': value['arguments'],
                      'id': f'{call_prefix}_{i}', 'type': 'tool_call'})
    if not calls and not raw.strip():
        raise ValueError('Empty final response')
    content = re.sub(r'<tool_call>.*?</tool_call>', '', raw, flags=re.S).strip()
    return AIMessage(content=content, tool_calls=calls)


def serialize(messages, schemas):
    """Keep native instructions; add Qwen function protocol at the transport boundary."""
    result = []
    pending = {}
    for message in messages:
        role = {'human': 'user', 'ai': 'assistant'}.get(message.type, message.type)
        if role == 'tool':
            expected = pending.pop(message.tool_call_id, None)
            if expected is None or expected != message.name:
                raise ValueError('Tool response does not match a prior call')
            result.append({'role': 'user', 'content': '<tool_response>\n' + json.dumps({
                'name': message.name, 'content': message.content}, ensure_ascii=False) + '\n</tool_response>'})
        elif role == 'assistant' and message.tool_calls:
            text = message.content or ''
            for call in message.tool_calls:
                if call['id'] in pending:
                    raise ValueError('Duplicate tool-call ID')
                pending[call['id']] = call['name']
                text += '\n<tool_call>\n' + json.dumps({'name': call['name'], 'arguments': call['args']}) + '\n</tool_call>'
            result.append({'role': role, 'content': text})
        else:
            result.append({'role': role, 'content': message.content})
    if pending:
        raise ValueError('Missing tool response')
    protocol = ('\n\n# Tools\nYou may call one or more functions to assist with the user query.\n'
                'Function signatures are inside <tools></tools> tags:\n<tools>\n' +
                '\n'.join(json.dumps(x, ensure_ascii=False) for x in schemas) +
                '\n</tools>\nFor each function call, return a JSON object with function name '
                'and arguments inside <tool_call></tool_call> tags:\n'
                '<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>')
    if result and result[0]['role'] == 'system':
        result[0]['content'] += protocol
    else:
        result.insert(0, {'role': 'system', 'content': protocol.strip()})
    return result


class LocalVisionModel:
    def __init__(self, model_path, limits, record, guard):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True,
            min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map='cuda:0',
            attn_implementation='sdpa', local_files_only=True).eval()
        self.limits, self.record, self.guard = limits, record, guard
        self.calls = self.tokens = 0
        self.deadline = 0
        self.schemas = []

    def bind_tools(self, tools):
        self.schemas = [{'type': 'function', 'function': {'name': t.name,
            'description': t.description, 'parameters': t.args_schema.model_json_schema()}}
            for t in tools]
        return self

    def start_episode(self):
        self.calls = self.tokens = 0
        self.deadline = time.time() + self.limits['episode_seconds']

    def invoke(self, messages):
        import torch
        from qwen_vl_utils import process_vision_info
        self.guard()
        if self.calls >= self.limits['planner_calls'] or time.time() >= self.deadline:
            raise RuntimeError('Episode planner/time budget exhausted')
        converted = serialize(messages, self.schemas)
        prompt = self.processor.apply_chat_template(converted, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(converted)
        inputs = self.processor(text=[prompt], images=images, videos=videos, padding=True,
                                return_tensors='pt').to(self.model.device)
        if not images or 'pixel_values' not in inputs or not inputs['pixel_values'].numel():
            raise RuntimeError('Actual vision payload is required')
        if inputs['input_ids'].shape[1] > self.limits['input_tokens']:
            raise RuntimeError('Context budget exceeded; no silent truncation')
        allowed = min(self.limits['output_tokens_per_call'],
                      self.limits['output_tokens_total'] - self.tokens)
        if allowed <= 0:
            raise RuntimeError('Episode output-token budget exhausted')
        self.calls += 1
        start = time.time()
        self.record('planner_started', {'call': self.calls, 'images': len(images),
            'pixel_values_shape': list(inputs['pixel_values'].shape),
            'image_grid_thw': inputs['image_grid_thw'].tolist(),
            'input_tokens': inputs['input_ids'].shape[1]})
        with torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, max_new_tokens=allowed,
                max_time=max(1, min(self.deadline - time.time(), 180)), use_cache=True)
        generated = output[0, inputs['input_ids'].shape[1]:]
        self.tokens += len(generated)
        raw = self.processor.decode(generated, skip_special_tokens=True)
        self.record('planner_finished', {'call': self.calls, 'output_tokens': len(generated),
            'seconds': time.time() - start, 'raw': raw})
        return parse_response(raw, f'call{self.calls}')
