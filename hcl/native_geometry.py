"""Opt-in Qwen2.5-VL schema compatibility for explicitly declared pixel-coordinate runs."""
import json
import math
from pathlib import Path
from .evaluator import _parse_jsonish
from .models import LocalTransformersModel, _state_phase

class NativePixelModel(LocalTransformersModel):
    def generate(self, prompt, *, state):
        raw = super().generate(prompt, state=state)
        if _state_phase(state) != 'final_answer':
            return raw
        context = state.get('task_context') or {}
        question = str(state.get('question') or context.get('raw_question') or prompt).lower()
        if 'return only one json object with bbox' not in question:
            return raw
        value = _parse_jsonish(raw)
        if not (isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict)):
            return raw
        box = value[0].get('bbox_2d', value[0].get('bbox'))
        if not (isinstance(box, list) and len(box) == 4):
            return raw
        try:
            box = [float(x) for x in box]
        except (TypeError, ValueError):
            return raw
        if not (all(math.isfinite(x) for x in box) and box[2] > box[0] and box[3] > box[1]):
            return raw
        canonical = json.dumps({'bbox': box})
        if self.config.log_path:
            with Path(self.config.log_path + '.schema_normalization.jsonl').open('a') as f:
                f.write(json.dumps({'task_id': state.get('task_id'), 'profile': 'native_pixel_single_bbox', 'raw': raw, 'canonical': canonical}) + '\n')
        return canonical
