import json
from run import compare, state, guidance
a = {"c":{"answer":"A","views":["1","2","3"]}}
b = {"c":{"answer":"B","views":["1","2","4"]}}
r = compare(a,b,{"c":"B"})
assert (r["gain"],r["net"],r["loss"],r["changed_image_sets"]) == (1,1,0,1)
assert guidance(state(),{})==state()["skill"]
print(json.dumps({"passed":True}))
