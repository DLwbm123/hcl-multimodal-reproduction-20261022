"""CPU-only checks; no model generation or scientific outcomes."""
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from native_runtime import NoUpdateHCL
from vision_transport import parse_response, serialize


def main(model_path):
    from transformers import AutoProcessor
    call = parse_response('<tool_call>{"name":"chest_xray_expert","arguments":{"image_paths":["x"]}}</tool_call>', 'c')
    messages = [SystemMessage(content='native prompt'), HumanMessage(content='question'), call,
                ToolMessage(content='observed result', tool_call_id='c_0', name='chest_xray_expert')]
    converted = serialize(messages, [])
    assert converted[0]['content'].startswith('native prompt')
    assert 'observed result' in converted[-1]['content']
    for raw in ['<tool_call>{}', '<tool_call>{"name":"x","arguments":"{}"}</tool_call>', '']:
        try:
            parse_response(raw, 'x')
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError('Malformed call accepted')
    try:
        serialize(messages[:-1] + [ToolMessage(content='x', tool_call_id='wrong', name='x')], [])
    except ValueError:
        pass
    else:
        raise AssertionError('Mismatched response accepted')
    class Fake:
        def bind_tools(self, tools):
            self.tools = tools
            return self
        def invoke(self, inputs):
            self.inputs = inputs
            return call
    model = Fake()
    wrapper = NoUpdateHCL(model).bind_tools(['sentinel'])
    assert wrapper.invoke(messages) is call and model.inputs is messages and model.tools == ['sentinel']
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True,
        min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / 'fixture.png'
        array = np.zeros((448, 448, 3), dtype=np.uint8)
        array[64:256, 64:256] = 255
        Image.fromarray(array).save(path)
        prompt = processor.apply_chat_template([{'role': 'user', 'content': [
            {'type': 'image', 'image': str(path)}, {'type': 'text', 'text': 'Describe image'}]}],
            tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[Image.open(path)], return_tensors='pt')
        assert inputs['pixel_values'].numel() > 0 and inputs['image_grid_thw'].shape[0] == 1
        assert '<|vision_start|>' in prompt and '<|image_pad|>' in prompt
        print('CPU_TRANSPORT_OK', list(inputs['pixel_values'].shape), 'model_generations=0', flush=True)


if __name__ == '__main__':
    import json, os
    root = Path(os.environ['RUN_ROOT'])
    main(json.loads((root / 'runtime.private.json').read_text())['planner_path'])
