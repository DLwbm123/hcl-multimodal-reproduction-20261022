"""Opt-in next-protocol draft. Never imported by the current experiment."""
from dataclasses import dataclass
import json
import math
import re


def checked_box(box, width, height):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError('one xyxy box required')
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in box):
        raise ValueError('finite real coordinates required')
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError('box outside declared frame or nonpositive extent')
    return list(box)


@dataclass(frozen=True)
class FrameContract:
    """Captured before harness rewriting; frame is never inferred from the output."""
    original_width: int
    original_height: int
    input_width: int
    input_height: int
    output_frame: str = 'input_pixel_half_open_xyxy'

    def __post_init__(self):
        for value in (self.original_width, self.original_height, self.input_width, self.input_height):
            if type(value) is not int or value <= 0:
                raise ValueError('positive integer dimensions required')
        if self.input_width % 28 or self.input_height % 28:
            raise ValueError('Qwen input dimensions must be multiples of 28')
        if self.output_frame != 'input_pixel_half_open_xyxy':
            raise ValueError('unknown or ambiguous output frame')

    def visible_contract(self):
        # No answer, mask, role, method, or private source metadata.
        return {'input_width': self.input_width, 'input_height': self.input_height,
                'bbox_coordinate_system': self.output_frame}

    def instruction(self):
        return (f'The supplied image is {self.input_width} pixels wide and {self.input_height} pixels high. '
                'Return one JSON object with bbox: [x1,y1,x2,y2] in the supplied-image pixel frame, '
                'half-open xyxy. Do not use normalized 0-1000 coordinates or original-file coordinates.')

    def to_input(self, original_box):
        box = checked_box(original_box, self.original_width, self.original_height)
        return [v / source * target for v, source, target in zip(
            box, (self.original_width, self.original_height) * 2,
            (self.input_width, self.input_height) * 2)]

    def to_original(self, input_box):
        box = checked_box(input_box, self.input_width, self.input_height)
        return [v / source * target for v, source, target in zip(
            box, (self.input_width, self.input_height) * 2,
            (self.original_width, self.original_height) * 2)]

    def normalize_answer(self, raw):
        text = raw.strip()
        fence = re.fullmatch(r'```(?:json)?\s*(.*?)\s*```', text, re.I | re.S)
        if fence:
            text = fence.group(1)
        value = json.loads(text)
        item = value[0] if isinstance(value, list) and len(value) == 1 else value
        if not isinstance(item, dict) or not set(item) <= {'bbox', 'bbox_2d', 'label'}:
            raise ValueError('ambiguous bbox object')
        if ('bbox' in item) == ('bbox_2d' in item):
            raise ValueError('exactly one bbox field required')
        box = item.get('bbox', item.get('bbox_2d'))
        # No clipping, guessing, rounding, multiple-box choice, or ground truth access.
        return json.dumps({'bbox': self.to_original(box)})

    def prepare_pixels(self, image, image_processor):
        """CPU only. Use returned tensors in the future caller; never resize twice."""
        if image.size != (self.original_width, self.original_height):
            raise ValueError('source image dimensions disagree with contract')
        supplied = image.convert('RGB').resize((self.input_width, self.input_height),
                                               resample=image_processor.resample)
        tensors = image_processor(images=[supplied], do_resize=False, return_tensors='pt')
        grid = tensors['image_grid_thw'].tolist()
        expected = [[1, self.input_height // image_processor.patch_size,
                      self.input_width // image_processor.patch_size]]
        if grid != expected:
            raise ValueError(f'processor frame mismatch: {grid} != {expected}')
        return tensors
