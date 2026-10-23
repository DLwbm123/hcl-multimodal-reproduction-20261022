"""Original AgentLego vision tools in their existing compatible environment."""
import ast
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(os.environ['RUN_ROOT'])


def validate(name, args, allowed_image):
    if name != 'Calculator':
        if Path(args.get('image', '')).resolve() != Path(allowed_image).resolve():
            raise ValueError('Image is outside this episode')
    else:
        tree = ast.parse(args['expression'], mode='eval')
        permitted = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Call,
                     ast.Name, ast.Load, ast.Attribute, ast.Add, ast.Sub, ast.Mult,
                     ast.Div, ast.Pow, ast.Mod, ast.FloorDiv, ast.UAdd, ast.USub)
        for node in ast.walk(tree):
            if not isinstance(node, permitted):
                raise ValueError('Unsupported calculator syntax')
            if isinstance(node, ast.Attribute) and not (isinstance(node.value, ast.Name) and node.value.id == 'math' and not node.attr.startswith('_')):
                raise ValueError('Only public math attributes allowed')
            if isinstance(node, ast.Name) and node.id.startswith('_'):
                raise ValueError('Private calculator names forbidden')


def main():
    channel = sys.stdout
    def send(value):
        channel.write(json.dumps(value, default=str) + '\n')
        channel.flush()
    with contextlib.redirect_stdout(sys.stderr):
        try:
            import torch
            import mmpretrain
            from mmengine.config import Config
            from mmpretrain.apis import VisualQuestionAnsweringInferencer
            from agentlego.tools.ocr.ocr import OCR
            from agentlego.tools.calculator.python_calculator import Calculator
            from agentlego.types import ImageIO
            spec = importlib.util.spec_from_file_location('native_benchmark', ROOT / 'upstream/agentlego/benchmark.py')
            native = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(native)
            paths = json.loads((ROOT / 'runtime.private.json').read_text())
            cfg = Config.fromfile(Path(mmpretrain.__file__).parent / '.mim/configs/llava/llava-7b-v1.5_vqa.py')
            cfg.model.tokenizer.name_or_path = paths['llava_tokenizer']
            cfg.model.lang_encoder.name_or_path = paths['llama_config']
            cfg.model.vision_encoder.pretrained = paths['clip_checkpoint']
            infer = VisualQuestionAnsweringInferencer(cfg, pretrained=paths['llava_checkpoint'], device='cuda:0')
            tools = {'ImageDescription': native.ImageDescription(),
                     'RegionAttributeDescription': native.RegionAttributeDescription(),
                     'OCR': OCR(device='cuda:0'), 'Calculator': Calculator()}
            for name in ['ImageDescription', 'RegionAttributeDescription']:
                tools[name]._inferencer = infer
                tools[name]._is_setup = True
            tools['OCR'].setup()
            tools['OCR']._is_setup = True
            costs = []
            current = {'request_id': None}
            def ledger(value):
                with (ROOT / 'vision_events.private.jsonl').open('a') as stream:
                    stream.write(json.dumps({**current, **value}) + '\n')
            original = infer.model.model.generate
            def measured(*args, **kwargs):
                start = time.time()
                pixels = kwargs.get('images')
                entry = {'status': 'started', 'pixel_shape': list(pixels.shape),
                         'input_tokens': int(args[0].shape[-1]), 'time': start}
                costs.append(entry)
                ledger(entry)
                with torch.inference_mode():
                    value = original(*args, **kwargs)
                entry.update(status='completed', seconds=time.time() - start,
                             output_tokens=int(value.shape[-1] - args[0].shape[-1]))
                ledger(entry)
                return value
            infer.model.model.generate = measured
            send({'ready': True, 'tools': list(tools)})
        except Exception as exc:
            send({'ready': False, 'error': repr(exc), 'traceback': traceback.format_exc()})
            return 1
        for line in sys.stdin:
            request = json.loads(line)
            if request is None:
                break
            current['request_id'] = request['request_id']
            costs.clear()
            start = time.time()
            try:
                guard = json.loads((ROOT / 'guard.json').read_text())
                if guard['stop'] or time.time() - guard['time'] > 90:
                    raise RuntimeError('Time/storage guard stopped or stale')
                name, args = request['name'], request['args']
                validate(name, args, request['allowed_image'])
                if name == 'RegionAttributeDescription':
                    from agentlego.utils import parse_multi_float
                    from PIL import Image
                    x1, y1, x2, y2 = [int(x) for x in parse_multi_float(args['bbox'])]
                    with Image.open(args['image']) as image:
                        width, height = image.size
                    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                        raise ValueError('Crop outside image bounds')
                result = tools[name](**args)
                send({'request_id': request['request_id'], 'ok': True, 'result': str(result), 'seconds': time.time() - start, 'generations': costs})
            except Exception as exc:
                send({'request_id': request['request_id'], 'ok': False, 'error': repr(exc), 'seconds': time.time() - start, 'generations': costs})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
