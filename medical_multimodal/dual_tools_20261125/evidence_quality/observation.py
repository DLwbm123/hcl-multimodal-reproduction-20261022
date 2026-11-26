"""Read visible evidence with the frozen visual model; never use reference data."""
import json
from PIL import Image,ImageDraw
from grounding import locate

LOCAL='RegionAttributeDescription'
GLOBAL='ImageDescription'


def region_view(image,box):
    x,y,X,Y=box
    assert 0<=x<X<=image.width and 0<=y<Y<=image.height
    overview=image.copy()
    ImageDraw.Draw(overview).rectangle((x,y,X-1,Y-1),outline=(0,220,255),width=max(1,round(min(image.size)/150)))
    crop=image.crop(box)
    sheet=Image.new('RGB',(900,450),'black')
    for i,panel in enumerate((overview,crop)):
        panel.thumbnail((450,450),Image.Resampling.LANCZOS)
        sheet.paste(panel,(i*450+(450-panel.width)//2,(450-panel.height)//2))
    return sheet


def observe(gen,q,image,box=None):
    instruction=('Document only visual evidence relevant to the requested fact. Begin with the most informative visible finding. '
                 'Identify the depicted structure or marked site and its observable appearance, location, or legible text/number as relevant. '
                 'For diagrams use the printed labels and their relations. Preserve orientation. If a needed feature cannot be seen, say so. '
                 'Do not supply a final answer or differential, restate the question, or add general medical background. At most 50 words.')
    view=image
    if box is not None:
        view=region_view(image,box)
        instruction+=' The left panel is the overview; cyan marks the crop. The right panel shows that crop. Localize with the overview and describe what is visible in the crop.'
    return gen('Clinical information needed: '+q+'\n'+instruction,96,view).strip()


def visual_tools(original,visible,choose,gen):
    def call(q,image,path,notes,name):
        if name not in (GLOBAL,LOCAL):return original(q,image,path,notes,name)
        box=None
        if name==LOCAL:
            used={tuple(json.loads(n['arg'])['box']) for n in notes if n['name']==LOCAL and n['ok']}
            box=locate(choose,visible(q,notes),image,used)
            if box is None:return dict(name=name,ok=False,text='No unused valid region.',arg='')
        arg=json.dumps({'box':box} if box else {},sort_keys=True)
        if any(n['name']==name and n['arg']==arg for n in notes):
            return dict(name=name,ok=False,text='Identical request already attempted; no new evidence.',arg=arg)
        text=observe(gen,q,image,box)
        return dict(name=name,ok=bool(text),text=text,arg=arg)
    return call
