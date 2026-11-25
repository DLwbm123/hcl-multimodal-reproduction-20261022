import json
from routing import choose, evaluate, fit
r={"available":True,"margin":.03,"agreement":.5,"default":{"score":.4,"success":False,"valid":True},"review":{"score":.8,"success":True,"valid":True}}
s=dict(r,margin=.18,agreement=.9,default=r["review"],review=r["default"])
assert choose("both",r) and not choose("both",s)
assert fit([r,s])=="low_margin"
assert evaluate([r,s],["low_margin"]*2)["mean_vlm_calls"]==1.5
print(json.dumps({"passed":True}))
