import json
from PIL import Image
from clinical import split
from grounding import boxes, iou, locate, reference_box
assert list(map(len,split()))==[30,16,45]
im=Image.new('RGB',(131,177)); menu=boxes(im.size)
assert all(0<=x<X<=131 and 0<=y<Y<=177 for x,y,X,Y in menu)
b=locate(lambda q,images,instr,opts: opts[0],'',im,{menu[0]})
assert b==menu[1] and iou(b,b)==1.
assert reference_box({'bbox':'(-1, 2, 50, 180)'},im.size)==((0,2,50,177),True)
assert reference_box({},im.size)==(None,False)
print(json.dumps({'passed':True}))
