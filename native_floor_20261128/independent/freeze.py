"""Reserve public sample IDs using the native-coordinate data constructor."""
import json
import os
from pathlib import Path

from scripts import build_multimodal_local as builder

SEED = 26112893


def main():
    root=Path(os.environ['REPRO_ROOT']).resolve()
    old=Path(os.environ['ORIGINAL'])
    histories=[json.loads((old/name).read_text()) for name in
               ('split_manifest.json','split_manifest_native_pixels_50_10_100.json')]
    used=[{int(row['image_id']) for splits in history['records'].values()
           for rows in splits.values() for row in rows} for history in histories]
    assert all(v for v in used)
    excluded=set.union(*used)
    builder.SEED=SEED

    def missing(*_args,**_kwargs):
        raise RuntimeError('image missing from the read-only original image store')

    builder.urllib.request.urlopen=missing
    (root/'code/configs').mkdir(parents=True,exist_ok=True)
    builder.build(counts={'train':0,'validation':0,'test':100},profile='p',
                  native_pixels=True,excluded_test_images=excluded)
    report=json.loads((root/'split_manifest_p.json').read_text())
    records={task:parts['test'] for task,parts in report['records'].items()}
    assert len(records)==4 and all(len(v)==len({r['image_id'] for r in v})==100 for v in records.values())
    selected={r['image_id'] for rows in records.values() for r in rows}
    assert not selected&excluded
    (root/'PANEL.json').write_text(json.dumps(dict(seed=SEED, records=records, distinct_images=len(selected),
                                excluded_image_ids=len(excluded), historical_manifest_counts=list(map(len,used))),
                                indent=2)+'\n')


if __name__=='__main__':
    main()
