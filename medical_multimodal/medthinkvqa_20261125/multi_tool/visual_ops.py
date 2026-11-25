"""Three pixel-returning operations with equal-sized, aspect-preserving observations."""
import string
from pathlib import Path

from PIL import Image, ImageOps

SIDE = 560
REGIONS = {
    "center":(.2,.2,.8,.8), "upper-left":(0.,0.,.6,.6), "upper-right":(.4,0.,1.,.6),
    "lower-left":(0.,.4,.6,1.), "lower-right":(.4,.4,1.,1.),
}
OPEN, CROP, COMPARE = "open_view","inspect_region","compare_regions"


def tile(im, width=SIDE):
    return ImageOps.pad(im,(width,SIDE),method=Image.Resampling.BICUBIC,color=(32,32,32))


def region(im, name):
    w,h = im.size
    x0,y0,x1,y1 = REGIONS[name]
    return im.crop((int(x0*w),int(y0*h),max(int(x1*w),int(x0*w)+1),max(int(y1*h),int(y0*h)+1)))


class VisualTools:
    def __init__(self, directory, case):
        self.folder = Path(directory)/"images"/case["case"]
        self.catalog = {v["view"]:v for v in case["views"]}
        self.raw, self.shown, self.trace = {}, {}, []

    def moves(self, name):
        if name == OPEN:
            return [(v,) for v in self.catalog if v not in self.raw]
        if name == CROP:
            return [(v,r) for v in self.raw for r in REGIONS if (CROP,(v,r)) not in self.trace]
        if name == COMPARE and len(self.raw)==2:
            left,right = self.raw
            return [(left,right,r) for r in REGIONS]
        return []

    def invoke(self, name, args):
        if len(self.trace)>=3 or args not in self.moves(name):
            raise ValueError("Unavailable pixel operation")
        if name == OPEN:
            path, = self.folder.glob(args[0]+".*")
            with Image.open(path) as source:
                self.raw[args[0]] = source.convert("RGB")
            out = tile(self.raw[args[0]])
        elif name == CROP:
            v,r = args
            out = tile(region(self.raw[v],r))
        else:
            left,right,r = args
            out = Image.new("RGB",(SIDE,SIDE))
            for i,v in enumerate((left,right)):
                out.paste(tile(region(self.raw[v],r),SIDE//2),(i*SIDE//2,0))
        self.trace.append((name,args))
        key = f"{len(self.trace)} {name}({', '.join(args)})"
        self.shown[key] = out


def multi_episode(choose,directory,case,fixed=False):
    t = VisualTools(directory,case)
    views = list(t.catalog)
    opts = case["options"]
    context = ("History: " + case["history"] + "\nPossible diagnoses:\n" +
               "\n".join(f"{k}. {v}" for k,v in opts.items()) + "\nAvailable study directory:\n" +
               "\n".join(f"{k}: {v['modality']}, {v['type']}" for k,v in t.catalog.items()))
    t.invoke(OPEN,(views[0],))
    for step in range(2):
        menu = {k:v for k,v in zip("ABC",(OPEN,CROP,COMPARE)) if t.moves(v)}
        instruction = ("Choose the most informative next operation. open_view shows a new full image; "
                       "inspect_region enlarges part of an opened original; compare_regions shows two "
                       "opened images' same relative region side by side, without anatomical registration. "
                       "Select the operation letter.\n" + "\n".join(f"{k}: {v}" for k,v in menu.items()))
        name = menu[choose(context,t.shown,instruction,list(menu))]
        if fixed:
            name = OPEN if step==0 else COMPARE
        args = t.moves(name)
        if not 1<=len(args)<=26:
            raise ValueError("argument menu capacity")
        menu_args = dict(zip(string.ascii_uppercase,args))
        select = "Select the arguments for "+name+". Return one letter.\n"+ "\n".join(
            f"{k}: {', '.join(v)}" for k,v in menu_args.items())
        args_chosen = menu_args[choose(context,t.shown,select,list(menu_args))]
        t.invoke(name,args[0] if fixed else args_chosen)
    answer = choose(context,t.shown,"Use the shown observations and history to choose the most likely diagnosis. "
                    "Return a single option letter.",list(opts))
    if len(t.shown)!=3 or any(x.size!=(SIDE,SIDE) for x in t.shown.values()) or answer not in opts:
        raise ValueError("invalid multimodal episode")
    return dict(answer=answer,views=list(t.raw),actions=[n for n,_ in t.trace],
                trace=[n+"("+",".join(a)+")" for n,a in t.trace])
