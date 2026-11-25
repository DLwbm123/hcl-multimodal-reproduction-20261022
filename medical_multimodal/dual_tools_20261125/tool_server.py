"""Line RPC for the four prepared MedCTA tools in their original Python environment."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import sys

reply = os.fdopen(os.dup(1),'w',buffering=1)
os.dup2(2,1)
base = Path(os.environ['CTA_ROOT'])
spec = importlib.util.spec_from_file_location('tool_definitions', base/'code/agentlego/benchmark.py')
defs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(defs)
from agentlego.tools import Calculator, OCR
from agentlego.types import ImageIO

def safe_expression(expr):
    if not 0 < len(expr) <= 90:
        return False
    try:
        nodes = list(ast.walk(ast.parse(expr, mode='eval')))
    except (ValueError,SyntaxError):
        return False
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd)
    return len(nodes) <= 45 and all(isinstance(n, allowed) for n in nodes) and all(
        type(n.value) in (int,float) and abs(n.value) <= 1e12 for n in nodes if isinstance(n, ast.Constant))

tools = {'Calculator':Calculator(),'OCR':OCR(device='cuda'),
         'ImageDescription':defs.ImageDescription(device='cuda'),
         'RegionAttributeDescription':defs.RegionAttributeDescription(device='cuda')}
for name in ('OCR','ImageDescription','RegionAttributeDescription'):
    tools[name].setup()
print('READY',file=reply)
for line in sys.stdin:
    arg=json.loads(line)
    name=arg['name']; tool=tools[name]
    try:
        if name=='Calculator':
            if not safe_expression(arg['expression']):raise ValueError('unsupported arithmetic expression')
            value=tool.apply(arg['expression'])
        else:
            image=ImageIO(arg['path'])
            if name=='RegionAttributeDescription':
                value=tool.apply(image,str(tuple(arg['box'])),arg['attribute'])
            else:
                value=tool.apply(image)
        print(json.dumps({'ok':True,'text':str(value)[:1600]}),file=reply)
    except (ValueError,ZeroDivisionError,SyntaxError) as e:
        print(json.dumps({'ok':False,'text':'Tool argument rejected: '+type(e).__name__}),file=reply)
