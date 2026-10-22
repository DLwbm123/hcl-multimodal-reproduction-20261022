"""Prepare deterministic datasets and private runtime configs; no model calls."""
import importlib.util,json,os,time
from pathlib import Path
R=Path(os.environ['REPRO_ROOT']).resolve();C=Path(__file__).resolve().parent
assert C==R/'code', 'Place this checkout at $REPRO_ROOT/code'
assert Path(os.environ['MODEL_PATH']).is_dir(), 'MODEL_PATH must reference an existing local model'
(R/'tmp').mkdir(exist_ok=True);(R/'cache').mkdir(exist_ok=True)
budget=R/'budget.json'
if not budget.exists():
 budget.write_text(json.dumps({'started_epoch':time.time(),'stop_after_seconds':84600,'limit_seconds':86400,'strict_limit_bytes':50000000000,'stop_at_allocated_bytes':47000000000},indent=2))
assert time.time()-json.loads(budget.read_text())['started_epoch']<84600, 'Existing budget expired; do not reset it for retries'
spec=importlib.util.spec_from_file_location('builder',C/'scripts/build_multimodal_local.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
b.build()
pilot=json.loads((R/'split_manifest.json').read_text())
excluded={x['image_id'] for t in pilot['records'].values() for x in t['test']}
b.build(counts={'train':50,'validation':10,'test':100},profile='native_pixels_50_10_100',native_pixels=True,excluded_test_images=excluded)
expected=json.loads((C/'results/sample_ids_native.json').read_text())
actual=json.loads((R/'split_manifest_native_pixels_50_10_100.json').read_text())
assert actual['records']==expected['records'], 'Dataset source/ordering differs from the published sample IDs'
for arm in ['raw_zero_shot','static_harness','stability_hcl','plasticity_hcl']:
 text=(C/f'configs/native_{arm}.template.json').read_text().replace('@MODEL_PATH@',os.environ['MODEL_PATH']).replace('@REPRO_ROOT@',str(R))
 (C/f'configs/runtime_{arm}.json').write_text(text)
print('Prepared published sample IDs and runtime configs; no model loaded.')
