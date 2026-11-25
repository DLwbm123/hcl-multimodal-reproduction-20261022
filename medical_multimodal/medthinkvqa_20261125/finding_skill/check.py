import json
from run import BASE,JUDGE
from patch_format import normalize
from upstream_skill import apply_patch_with_report
patch,_=normalize('{"edits":[{"op":"append","content":"Check both sides where visible."}]}',BASE)
s,log=apply_patch_with_report(BASE,patch)
assert s!=BASE and log[0]["status"]=="applied_append" and "A = covered" in JUDGE
print(json.dumps({"passed":True}))
