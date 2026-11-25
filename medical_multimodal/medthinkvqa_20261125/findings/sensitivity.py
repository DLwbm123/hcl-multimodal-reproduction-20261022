"""Descriptive defined-before-unblinding subset and attribution summaries."""
import json
import os
from collections import Counter
from pathlib import Path
from panel import build
from run import paired

root=Path(os.environ["RUN_ROOT"])
rows=json.loads((root/"private_predictions.json").read_text())
cases,gold=build(Path(os.environ["RAW_ROOT"]),[Path(p) for p in os.environ["PRIOR_RUNS"].split(os.pathsep)])
subset=[]
for k,c in cases.items():
    descriptions=[v["caption"].strip() for v in c["images"] if v["caption"].strip()]
    if len(descriptions)==len(set(descriptions)):
        subset.append(k)
def compare(a,b):
    return paired([rows[k]["pred"][a] for k in subset],
                  [rows[k]["pred"][b] for k in subset],[gold[k] for k in subset])
groups={}
for k,row in rows.items():
    a=row["pred"]["self"]==gold[k]
    b=row["pred"]["checked"]==gold[k]
    group="both_correct" if a and b else "rescued" if b else "lost" if a else "both_wrong"
    c=groups.setdefault(group,Counter())
    c["cases"]+=1
    c["with_omission_or_contradiction"]+=any(x in {"omitted_or_insufficient","contradicted"} for x in row["coverage"])
print(json.dumps(dict(unique_reference_cases=len(subset),unique_references={
    b+"_vs_"+a:compare(a,b) for a,b in [("direct","checked"),("direct","self"),("self","checked")]},
    self_to_checked=groups),indent=2))
