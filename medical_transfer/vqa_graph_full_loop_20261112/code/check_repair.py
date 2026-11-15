import os,sys,json,tempfile
from pathlib import Path
from PIL import Image
R=Path(os.environ['RUN_ROOT']);sys.path.insert(0,str(R/'code'))
from vision_tools import VisionTools
class Backend:
 calls=0
 def generate(self,prompt,*,state):self.calls+=1;return 'fixture'
with tempfile.TemporaryDirectory(dir=R/'tmp') as tmp:
 d=Path(tmp);image=d/'input.png';Image.new('RGB',(100,100)).save(image);runtime=VisionTools(d/'tools');b=Backend();state={'task_context':{'files':[{'type':'image','path':str(image)}]}}
 result=runtime.execute('inspect_region',{'bbox':[0,0,13,47],'question':'inspect'},state,b)
 assert result['status']=='error' and result['tool_inference_executed'] is False and b.calls==0
 assert not list((d/'tools').glob('crop*'))
 result=runtime.execute('inspect_region',{'bbox':[0,0,28,28],'question':'inspect'},state,b)
 assert result['status']=='ok' and b.calls==1
 class Failure:
  def generate(self,*a,**k):raise RuntimeError('backend failure')
 try:runtime.execute('inspect_region',{'bbox':[0,0,28,28],'question':'inspect'},state,Failure())
 except RuntimeError as e:assert str(e)=='backend failure'
 else:raise AssertionError('backend failure swallowed')
result={'passed':True,'checks':['small crop becomes error observation without inference or artifact','legal crop still runs','backend runtime failures still propagate'],'scripted_only':True,'confirmation_answers_not_read':True}
(R/'REPAIR_CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
