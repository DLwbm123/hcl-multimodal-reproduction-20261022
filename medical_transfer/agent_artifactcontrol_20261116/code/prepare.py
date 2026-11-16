import os,json
from pathlib import Path
R=Path(os.environ['RUN_ROOT']);P=json.loads((R/'protocol.json').read_text());source=Path(P['source_run'])
assert not (R/'LAUNCHED.json').exists()
assert json.loads((source/'SUPERVISOR_EXIT.json').read_text())['success']
train=json.loads((source/'private_learning.json').read_text())['train']
assert len(train)==48 and len({e['task_id'] for e in train})==48 and all(e['split']=='train' for e in train)
(R/'private_learning.json').write_text(json.dumps({'train':train}))
assert not (R/'private_confirmation.json').exists()
(R/'DATA_CHECK.json').write_text(json.dumps({'training48_only':True,'no_selection_or_confirmation_copied':True,'model_calls':0,'members_unchanged':True}))
