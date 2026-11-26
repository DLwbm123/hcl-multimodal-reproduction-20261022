import json
from pathlib import Path
r=json.loads(Path('private_rubrics.json').read_text())
assert len(r)==45 and all(x['rule'] and x['question'] and x['reference'] for x in r.values())
print(json.dumps({'passed':True,'rubrics':len(r)}))
