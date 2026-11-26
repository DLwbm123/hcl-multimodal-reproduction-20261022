import json, os
from pathlib import Path
rows=json.loads((Path(os.environ['PANEL_ROOT'])/'inputs.json').read_text())
assert len(rows)==200 and all(Path(r['path']).is_file() for r in rows)
print(json.dumps({'passed':True,'n':len(rows)}))
