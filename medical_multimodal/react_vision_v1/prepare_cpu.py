"""Canonicalize single-frame inputs once for every arm; no model execution."""
import json
import os
from pathlib import Path
import time

from PIL import Image
import numpy as np
import skimage.io


def main():
    root = Path(os.environ['RUN_ROOT'])
    plan = json.loads((root / 'code/protocol.json').read_text())
    (root / 'canonical').mkdir(exist_ok=True)
    records = []
    for item in plan['panel']:
        source = root / 'data/chestagentbench/images' / item['images'][0]
        target = root / 'canonical' / (item['full_question_id'] + '.png')
        with Image.open(source) as image:
            if getattr(image, 'n_frames', 1) != 1:
                raise RuntimeError('Selected image has multiple frames; no silent first-frame selection')
            pixels = np.asarray(image.convert('RGB'))
            Image.fromarray(pixels).save(target)
            decoded = skimage.io.imread(target)
            # A single pixel check is warranted for the observed GIF dimensionality defect.
            assert decoded.ndim == 3 and decoded.shape[-1] == 3
            assert np.array_equal(pixels, decoded)
            records.append({'id': item['full_question_id'], 'source_format': image.format,
                            'shape': list(decoded.shape), 'source_preserved': True})
    (root / 'CANONICAL_INPUTS.json').write_text(json.dumps({'time': time.time(), 'images': records,
        'model_generations': 0, 'all_arms_share_files': True}, indent=2) + '\n')
    print('CANONICAL_INPUTS_OK:', len(records), 'model_generations=0')


if __name__ == '__main__':
    main()
