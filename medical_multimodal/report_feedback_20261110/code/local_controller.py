"""Local vision adapter for the unmodified upstream MedRAX Agent graph."""
import json
import os
import time
import uuid


def parse_action(text, allowed, final_only=False):
    value = text.strip()
    if final_only and value in tuple('ABCDEF'):
        return value, []
    if value.startswith('```') and value.endswith('```'):
        value = value.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    try:
        obj = json.loads(value)
    except json.JSONDecodeError:
        # Native MedRAX permits an ordinary textual final response. Never infer a
        # tool call from prose, and never rescue malformed tool JSON as prose.
        if not final_only and not any(s in value for s in ['"tool_calls"', '"args"', '<tool_call>']):
            return value, []
        raise
    if not isinstance(obj, dict) or set(obj) != {'content', 'tool_calls'}:
        raise ValueError('Controller must return content and tool_calls')
    if not isinstance(obj['content'], str) or not isinstance(obj['tool_calls'], list):
        raise ValueError('Invalid action types')
    calls = []
    for call in obj['tool_calls']:
        if not isinstance(call, dict) or set(call) != {'name', 'args'}:
            raise ValueError('Invalid tool call fields')
        if call['name'] not in allowed or not isinstance(call['args'], dict):
            raise ValueError('Unknown tool or malformed arguments')
        calls.append({'id': uuid.uuid4().hex, 'name': call['name'], 'args': call['args'], 'type': 'tool_call'})
    if len(calls) > 8:
        raise ValueError('At most eight tool calls per turn')
    return obj['content'], calls


class LocalController:
    model = None
    processor = None

    def __init__(self, plan, guard, record):
        self.plan, self.guard, self.record = plan, guard, record
        self.calls, self.deadline, self.tools = 0, 0, []

    def bind_tools(self, tools):
        self.tools = [{'name': t.name, 'description': t.description,
                       'parameters': t.args_schema.model_json_schema()} for t in tools]
        return self

    def load(self):
        if type(self).model is None:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            path = os.environ['LOCAL_MODEL_PATH']
            self.record('controller_loading', {})
            type(self).processor = AutoProcessor.from_pretrained(path, local_files_only=True,
                max_pixels=self.plan['controller']['max_pixels'])
            type(self).model = Qwen2_5_VLForConditionalGeneration.from_pretrained(path,
                torch_dtype=torch.bfloat16, device_map='cuda:0', attn_implementation='sdpa', local_files_only=True).eval()
            self.model.requires_grad_(False)
            self.record('controller_loaded', {})

    def invoke(self, messages):
        import torch
        from PIL import Image
        from langchain_core.messages import AIMessage
        self.guard()
        self.calls += 1
        if self.calls > self.plan['limits']['per_case_controller_calls'] or time.time() > self.deadline:
            raise RuntimeError('Episode controller budget exhausted')
        self.load()
        schema = ('Return exactly one JSON object with keys content (string) and tool_calls (list). '
                  'Each call has name (string) and args (object). No markdown. For a final answer use '
                  'an empty tool_calls list. Choose at most eight calls per turn; all calls count toward the case limit of eight. Never invent tool observations. '
                  'When asked to return only a choice letter, place only that letter in content and use no calls. '
                  'Available tool schemas: ' + json.dumps(self.tools))
        systems = [m.content for m in messages if m.type == 'system']
        transcript, images = [], []
        for m in messages:
            if m.type == 'system':
                continue
            value = {'role': m.type, 'content': m.content}
            if isinstance(m.content, list):
                parts = []
                for part in m.content:
                    if part['type'] == 'image_url':
                        path = part['image_url']['url']
                        with Image.open(path) as im:
                            images.append(im.convert('RGB').copy())
                        parts.append({'image_index': len(images), 'path': path})
                    else:
                        parts.append(part)
                value['content'] = parts
            if getattr(m, 'tool_calls', None):
                value['tool_calls'] = m.tool_calls
            if m.type == 'tool':
                value.update(name=m.name, tool_call_id=m.tool_call_id)
            transcript.append(value)
        content = [{'type': 'image', 'image': im} for im in images]
        content.append({'type': 'text', 'text': 'Conversation in order; images correspond to image_index:\n' + json.dumps(transcript)})
        chat = [{'role': 'system', 'content': '\n'.join(systems)+'\n'+schema}, {'role': 'user', 'content': content}]
        text = self.processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        kwargs = {'text': [text], 'return_tensors': 'pt'}
        if images:
            kwargs['images'] = images
        inputs = self.processor(**kwargs).to(self.model.device)
        length = int(inputs.input_ids.shape[-1])
        if length > self.plan['controller']['max_input_tokens']:
            raise ValueError('Controller input exceeds frozen token budget')
        start = time.time()
        self.record('controller_started', {'input_tokens': length, 'images': len(images)})
        with torch.inference_mode():
            result = self.model.generate(**inputs, max_new_tokens=self.plan['controller']['max_new_tokens'], do_sample=False,
                max_time=max(1, min(self.plan['limits']['request_timeout_seconds'], self.deadline-time.time())))
        output = self.processor.batch_decode(result[:, length:], skip_special_tokens=True)[0].strip()
        self.record('controller_finished', {'seconds': time.time()-start, 'output': output,
            'input_tokens': length, 'output_tokens': int(result.shape[-1])-length, 'image_count': len(images)})
        from policy import FINAL
        final_only = messages[-1].type == 'human' and messages[-1].content == FINAL
        body, calls = parse_action(output, {x['name'] for x in self.tools}, final_only=final_only)
        return AIMessage(content=body, tool_calls=calls)


if __name__ == '__main__':
    assert parse_action('Ordinary answer text.', set()) == ('Ordinary answer text.', [])
    assert parse_action('D', set(), final_only=True) == ('D', [])
    assert parse_action('{"content":"B","tool_calls":[]}', set()) == ('B', [])
    for text in ['{"content":"x","tool_calls":[{"name":"bad","args":{}}]}', '{"tool_calls": broken}']:
        try: parse_action(text, set())
        except (ValueError, TypeError): pass
        else: raise AssertionError(text)
    try: parse_action('Answer is D', set(), final_only=True)
    except ValueError: pass
    else: raise AssertionError('Final choice was guessed from prose')
    print('native text and tool boundaries passed')
