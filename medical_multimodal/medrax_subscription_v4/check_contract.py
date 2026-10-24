"""CPU check of native call/result IDs, per-case memory and the frozen image panel."""
import json
import os
from pathlib import Path
import sys
ROOT=Path(os.environ['R981_ROOT'])
sys.path.insert(0,str(ROOT/'code'))
from native_runtime import load_module
from langchain_core.messages import AIMessage,HumanMessage,ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from PIL import Image

Agent=load_module(ROOT/'upstream/medrax/agent/agent.py','contract_agent').Agent
class Tool:
    name='fixture';description='test only'
    def invoke(self,args):assert args=={'value':73};return {'receipt':'fixture73'}
class Model:
    calls=0
    def bind_tools(self,tools):return self
    def invoke(self,messages):
        self.calls+=1
        if self.calls==1:return AIMessage(content='',tool_calls=[{'id':'id73','name':'fixture','args':{'value':73},'type':'tool_call'}])
        results=[m for m in messages if isinstance(m,ToolMessage)]
        assert len(results)==1 and results[0].tool_call_id=='id73' and results[0].name=='fixture' and 'fixture73' in results[0].content
        return AIMessage(content='A' if self.calls==3 else 'received')
m=Model();agent=Agent(m,[Tool()],system_prompt='fixture',checkpointer=MemorySaver(),log_tools=False)
cfg={'configurable':{'thread_id':'fixture'}}
r=agent.workflow.invoke({'messages':[HumanMessage(content='start')]},cfg);assert r['messages'][-1].content=='received'
r=agent.workflow.invoke({'messages':[HumanMessage(content='letter')]},cfg);assert r['messages'][-1].content=='A' and m.calls==3
panel=json.loads((ROOT/'private/panel.json').read_text()); assert len(panel)==12 and len({r['case_id'] for r in panel})==12
images=[]
for row in panel:
 for name in row['images']:
  p=ROOT/'data/chestagentbench/images'/name
  with Image.open(p) as im:
   im.load();assert getattr(im,'n_frames',1)==1
   images.append({'size':im.size,'mode':im.mode})
result={'passed':True,'native_graph_calls':m.calls,'matching_tool_result_id':True,'second_turn_memory':True,'panel_cases':len(panel),'images':len(images),'dimensions':images}
(ROOT/'CPU_CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
