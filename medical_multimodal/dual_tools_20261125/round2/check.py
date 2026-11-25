import json, os
from adapt import append
assert append('Observe.','Inspect recorded evidence.')!='Observe.'
if os.environ['STUDY']=='med':
    from clinical import data,split
    assert list(map(len,split()))==[30,16,45]
else:
    from retina import split
    assert list(map(len,split()))==[120,80,40,160]
print(json.dumps({'passed':True}))
