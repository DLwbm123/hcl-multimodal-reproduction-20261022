import json
from decision import decide, training_summary
r = {"first_valid": True, "review_valid": True, "review_calls": 1, "stop": .3, "review": .8,
     "cues": {"area": .1, "margin": .2, "visual": "UNSURE"}}
s = dict(r, stop=.8, review=.3, cues={"area": .3, "margin": .2, "visual": "TIGHT"})
info = training_summary([r,s,r,s])
assert info["open_development"] and info["frozen_policy"] == "small_box"
assert info["loo_metrics"]["success"] == 4 and info["headroom"] > .2
assert not decide("review", dict(r,first_valid=False)) and not decide("visual",r)
print(json.dumps({"passed":True}))
