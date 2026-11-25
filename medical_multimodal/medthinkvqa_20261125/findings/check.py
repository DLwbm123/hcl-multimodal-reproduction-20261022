import json
from run import verify_span, paired
assert verify_span("Focal opacity in the left lung.","Focal opacity in the left lung.",["COPD"]) is None
assert verify_span("There is chronic obstructive pulmonary disease (COPD).","COPD",["Chronic obstructive pulmonary disease"])
assert verify_span("No mass in left lung.","mass in left lung",["other"])
assert paired(["A","C"],["B","C"],["B","C"])["gain"]==1
print(json.dumps({"passed":True}))
