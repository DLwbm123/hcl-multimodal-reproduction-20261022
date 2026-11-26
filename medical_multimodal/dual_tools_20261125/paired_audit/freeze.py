"""Freeze source criteria and private fact-specific adjudication notes before replay."""
import json
import os
from pathlib import Path

root = Path(os.environ['RUN_ROOT'])
data = Path(os.environ['DATA_ROOT'])
cases = {str(c['case_id']): c for c in map(json.loads, (data/'tasks.jsonl').read_text().splitlines())}
refs = json.loads((Path(os.environ['CTA_ROOT'])/'data/dataset.json').read_text())
ids = sorted(json.loads((Path(os.environ['QUALITY_RUN'])/'private_test.json').read_text()))
notes = (root/'private_notes.txt').read_text().splitlines()
assert len(ids) == len(notes) == 45
rubrics = {k: dict(question=cases[k]['question'], reference=refs[k]['gt_answer'], rule=rule)
           for k, rule in zip(ids, notes)}
with (root/'private_rubrics.json').open('x') as f:
    json.dump(rubrics, f, indent=2)
