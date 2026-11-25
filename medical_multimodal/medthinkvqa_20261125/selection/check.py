import json
import os
from pathlib import Path
from run import episode, validate

data = Path(os.environ["DATA_ROOT"]) / "development"
case = json.loads((data / "inputs.jsonl").open().readline())
def last(context, opened, instruction, labels):
    return labels[-1]
row = {name:episode(last,data,case,name=="adaptive") for name in ("fixed","adaptive")}
validate(row,[v["view"] for v in case["views"]])
assert row["fixed"]["receipts"][1] != row["adaptive"]["receipts"][1]
print(json.dumps({"passed":True,"check":"distinct actions route to distinct decoded images on development input"}))
