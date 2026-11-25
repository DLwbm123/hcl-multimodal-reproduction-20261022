import json
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw
from visual_ops import VisualTools, OPEN, CROP, COMPARE, multi_episode


with tempfile.TemporaryDirectory() as d:
    root=Path(d)
    images=root/"images"/"fixture"
    images.mkdir(parents=True)
    case=dict(case="fixture", history="test", options={"A":"x","B":"y"},
              views=[dict(view=v,type="test",modality="test") for v in "1234"])
    for i,key in enumerate("1234"):
        im=Image.new("RGB",(100,100),("red","green","blue","yellow")[i])
        ImageDraw.Draw(im).rectangle((0,0,49,49),fill="white")
        im.save(images/(key+".png"))
    t=VisualTools(root,case)
    t.invoke(OPEN,("1",))
    t.invoke(CROP,("1","upper-left"))
    assert len(t.raw)==1 and len(t.shown)==2
    assert list(t.shown.values())[0].getpixel((400,400)) != list(t.shown.values())[1].getpixel((400,400))
    t=VisualTools(root,case)
    t.invoke(OPEN,("1",))
    t.invoke(OPEN,("2",))
    t.invoke(COMPARE,("1","2","center"))
    im=list(t.shown.values())[-1]
    assert im.getpixel((210,320)) != im.getpixel((490,320))
    def choose(context,opened,instruction,labels):
        return labels[-1]
    assert multi_episode(choose,root,case,True)["actions"] == [OPEN,OPEN,COMPARE]
    assert multi_episode(choose,root,case)["actions"] == [OPEN,CROP,CROP]
print(json.dumps({"passed":True,"operations":3,"budget_checked":True}))
