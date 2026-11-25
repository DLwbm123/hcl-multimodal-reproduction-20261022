import json, os
if os.environ["STUDY"] == "retina":
    import retina
    from PIL import Image
    from pathlib import Path
    parts=retina.split()
    assert list(map(len,parts))==[120,80,40,160] and len(set(sum(parts,[])))==400
    data=retina.source(parts[0][0]); mask=retina.target(parts[0][0])
    assert mask.max()==2 and 0<retina.ratio(mask>0,mask==2)<=1 and retina.features(data)[0][0].shape[:2]==mask.shape
else:
    import clinical
    parts=clinical.split()
    assert len(set(sum(parts,[])))==91 and list(map(len,parts))==[30,16,45]
print(json.dumps({"passed":True}))
