import json
from collections import Counter
from run import pass_gate
a = Counter(checked=8, checked_cases=4, rescue=2, loss=1, case_gain=1)
assert pass_gate(a)
a['unsupported_right'] = 1
assert not pass_gate(a)
print(json.dumps({"passed": True}))
