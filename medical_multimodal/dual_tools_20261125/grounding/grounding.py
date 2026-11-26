"""Select an executable view on the raw image; reference geometry is used only for scoring."""
import ast
import json
from statistics import mean
from PIL import Image, ImageDraw

# Nine overlapping windows plus a wider central view. Coordinates are fractions of the source.
WINDOWS = [(x, y, x+.5, y+.5) for y in (0,.25,.5) for x in (0,.25,.5)] + [(.10,.10,.90,.90)]
LABELS = 'ABCDEFGHIJ'


def boxes(size):
    w,h = size
    return [tuple(round(v*s) for v,s in zip(coords,(w,h,w,h))) for coords in WINDOWS]


def locate(choose, context, image, excluded):
    choices = dict(zip(LABELS, boxes(image.size)))
    available = [k for k,b in choices.items() if b not in excluded and b[0]<b[2] and b[1]<b[3]]
    if not available:
        return None
    sheet = Image.new('RGB',(5*144,2*158),'white')
    draw = ImageDraw.Draw(sheet)
    for i,k in enumerate(LABELS):
        if k not in available:
            continue
        preview = image.crop(choices[k])
        preview.thumbnail((144,136),Image.Resampling.LANCZOS)
        x,y = (i%5)*144,(i//5)*158
        draw.text((x+4,y+1),k,fill='black')
        sheet.paste(preview,(x+(144-preview.width)//2,y+19))
    letter = choose(context,{'overview':image,'windows':sheet},
                    'Choose one letter from the labeled cropped views that best isolates the requested finding or measurement. '
                    'Use its visible contents. Available letters: '+' '.join(available),available)
    return choices[letter]


def reference_box(arguments, size):
    args = json.loads(arguments) if isinstance(arguments,str) else arguments
    try:
        raw = ast.literal_eval(args['bbox']) if isinstance(args['bbox'],str) else args['bbox']
        if len(raw)!=4 or not all(type(v) in (float,int) for v in raw):
            return None,False
        w,h = size
        clipped=tuple(max(0,min(v,s)) for v,s in zip(raw,(w,h,w,h)))
        if clipped[0]>=clipped[2] or clipped[1]>=clipped[3]:
            return None,False
        return clipped,tuple(raw)!=clipped
    except (KeyError,TypeError,ValueError,SyntaxError):
        return None,False


def iou(a,b):
    common=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    def area(box):
        return (box[2]-box[0])*(box[3]-box[1])
    return common/(area(a)+area(b)-common)


def summarize(rows):
    valid=[r for r in rows if r['iou'] is not None]
    return dict(n=len(rows),nonempty=sum(r['usable'] for r in rows),geometry_n=len(valid),
                clipped=sum(r['clipped'] for r in valid),
                mean_iou=mean(r['iou'] for r in valid) if valid else None,
                mean_center_iou=mean(r['center'] for r in valid) if valid else None,
                mean_menu_best_iou=mean(r['best'] for r in valid) if valid else None,
                iou_at_50=sum(r['iou']>=.5 for r in valid),
                before=sum(r['before'] for r in rows),after=sum(r['after'] for r in rows),
                gains=sum(r['after']>r['before'] for r in rows),losses=sum(r['after']<r['before'] for r in rows))
