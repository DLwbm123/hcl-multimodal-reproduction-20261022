"""Synthetic acceptance only. Never loads the 32 benchmark cases or their gold."""
import asyncio
import json
import os
import time
from pathlib import Path
from adapter import load_upstream,adapter_class
from tools import RestrictedTools,COLUMNS


def rows(values):
    return [dict(zip(COLUMNS,(patient,field,str(value),'2026-01-01',episode,unit,sample,'synthetic'))) for patient,field,value,episode,unit,sample in values]

CASES=[
 {'id':'synthetic_bmi','prompt':'For patient P-001, episode_000, compute Body Mass Index (BMI) as weight in kg divided by height in metres squared. Read the facts, call calculate, then submit the named result. No other calculator is requested.',
  'rows':rows([('P-001','weight',70,'episode_000','kg','Chart'),('P-001','height',1.75,'episode_000','m','Chart'),('OTHER','weight',35,'episode_000','kg','Chart'),('P-001','weight',154.3,'episode_lb','lb','Chart')]),
  'script':[('read_facts',{}),('calculate',{'expression':'w/(h**2)','variables':{'w':70,'h':1.75}}),('submit_answer',{'calculators':[{'name':'Body Mass Index (BMI)','episode_id':'episode_000','value':70/1.75**2}]})],
  'gold':{'calculator_answers':[{'name':'Body Mass Index (BMI)','final_answer':70/1.75**2,'inputs':[]}]}},
 {'id':'synthetic_chain','prompt':'For patient P-002, episode_000, compute Synthetic Concentration Ratio = serum numerator / serum denominator. Then compute Synthetic Scaled Quantity = that ratio times scale. These are synthetic arithmetic exercises, not clinical formulas. Read current facts, calculate both outputs, and submit both full names. Ignore urine and other-patient rows.',
  'rows':rows([('P-002','numerator',8,'episode_000','mg/dL','Serum'),('P-002','denominator',2,'episode_000','mg/dL','Serum'),('P-002','scale',3,'episode_000','1','Chart'),('P-002','numerator',80,'episode_urine','mg/dL','Urine'),('OTHER','numerator',999,'episode_000','mg/dL','Serum')]),
  'script':[('read_facts',{}),('calculate',{'expression':'n/d','variables':{'n':8,'d':2}}),('calculate',{'expression':'ratio*scale','variables':{'ratio':4,'scale':3}}),('submit_answer',{'calculators':[{'name':'Synthetic Concentration Ratio','episode_id':'episode_000','value':4},{'name':'Synthetic Scaled Quantity','episode_id':'episode_000','value':12}]})],
  'gold':{'calculator_answers':[{'name':'Synthetic Concentration Ratio','final_answer':4,'inputs':[]},{'name':'Synthetic Scaled Quantity','final_answer':12,'inputs':[{'field':'Synthetic Concentration Ratio','value':4}]}]}}
]

class ScriptedModel:
    def __init__(self,actions):self.actions=list(actions);self.views=[]
    def bind_tools(self,schemas): self.schemas=schemas;return self
    async def ainvoke(self,messages):
        from langchain_core.messages import AIMessage
        self.views.append(messages)
        name,args=self.actions.pop(0)
        return AIMessage(content='',tool_calls=[{'name':name,'args':args,'id':f'call-{len(self.views)}','type':'tool_call'}])

class LocalModel:
    """Explicit JSON transport adapter; native provider tool-calling is not claimed."""
    def __init__(self,backend,root):self.backend=backend;self.root=root;self.calls=0;self.schemas=[]
    def bind_tools(self,schemas):self.schemas=schemas;return self
    async def ainvoke(self,messages):
        from langchain_core.messages import AIMessage
        budget=json.loads((self.root/'budget.json').read_text())
        guard=json.loads((self.root/'guard_status.json').read_text())
        if time.time()-guard['time']>30 or (self.root/'HARD_STOP.json').exists():raise RuntimeError('guard unavailable')
        if time.time()>=budget['early_stop_epoch'] or self.calls+budget.get('previous_model_calls',0)>=96:raise RuntimeError('model budget reached')
        self.calls+=1
        (self.root/'attempts.json').write_text(json.dumps({'calls':self.calls+budget.get('previous_model_calls',0)}))
        import torch
        # Use the checkpoint's actual multi-turn chat template. It has no native
        # function-call schema, so the explicit JSON contract remains disclosed.
        chat=[]
        for m in messages:
            role={'human':'user','ai':'assistant'}.get(m.type,m.type)
            content=m.content
            calls=getattr(m,'tool_calls',None)
            if calls:
                content=json.dumps({'tool_calls':[{'name':c['name'],'arguments':c['args']} for c in calls]})
            if role=='tool':
                content=json.dumps({'tool':getattr(m,'name',None),'observation':m.content,'status':getattr(m,'status',None)})
            chat.append({'role':role,'content':content})
        contract='\nReturn one complete JSON object: {"tool_calls":[{"name":"tool_name","arguments":{}}]} or {"final":"text"}. No markdown. Tool schemas: '+json.dumps(self.schemas)
        chat[0]['content']+=contract
        start=time.time()
        self.backend._load()
        tokenizer=self.backend._processor.tokenizer
        prompt=tokenizer.apply_chat_template(chat,tokenize=False,add_generation_prompt=True)
        encoded=tokenizer(prompt,return_tensors='pt').to(self.backend._model.device)
        if encoded['input_ids'].shape[-1]+1024>16384:raise RuntimeError('context budget')
        with torch.inference_mode():
            generated=self.backend._model.generate(**encoded,max_new_tokens=1024,do_sample=False)
        text=tokenizer.decode(generated[0,encoded['input_ids'].shape[-1]:],skip_special_tokens=True).strip()
        with (self.root/'model_calls.jsonl').open('a') as f:f.write(json.dumps({'call':self.calls,'seconds':time.time()-start,'output':text})+'\n')
        try:
            obj=json.loads(text)
        except json.JSONDecodeError:
            # Preserve malformed generation and return a normal failed tool observation.
            return AIMessage(content='',tool_calls=[{'name':'invalid_json_response','args':{},'id':f'{self.calls}-invalid','type':'tool_call'}])
        if type(obj) is not dict:raise ValueError('model response is not object')
        if set(obj)=={'final'} and type(obj['final']) is str:return AIMessage(content=obj['final'])
        if set(obj)!={'tool_calls'} or type(obj['tool_calls']) is not list or not 1<=len(obj['tool_calls'])<=8:raise ValueError('invalid model tool envelope')
        calls=[]
        for i,c in enumerate(obj['tool_calls']):
            if type(c) is not dict or set(c)!={'name','arguments'} or type(c['name']) is not str or type(c['arguments']) is not dict:raise ValueError('invalid call envelope')
            calls.append({'name':c['name'],'args':c['arguments'],'id':f'{self.calls}-{i}','type':'tool_call'})
        return AIMessage(content='',tool_calls=calls)

async def episode(upstream,root,case,model):
    out=root/case['id'];out.mkdir()
    prompt=out/'prompt.txt';prompt.write_text(case['prompt'])
    args=upstream.parse_args(['--workspace',str(out),'--prompt-file',str(prompt),'--model','frozen-local','--upstream','http://127.0.0.1:1/v1','--planning','on','--context-policy','none','--verification','off','--max-tokens','1024','--context-window','16384','--timeout-seconds','600','--seed','42'])
    manifest={};tools=RestrictedTools(case['rows'])
    h=adapter_class(upstream)(args,'http://127.0.0.1:1/v1',out,upstream.EventLog(out),manifest,tools=tools,model=model)
    await h.setup_tools(None)
    final=await h.build_graph().ainvoke({'turn':0,'termination':None,'final_text':None},config={'recursion_limit':110})
    result={'termination':final.get('termination'),'answer':tools.answer,'requests':h.requests,'tool_calls':tools.calls,'tool_names':manifest['tool_names'],'saw_facts':any(x.get('tool_name')=='read_facts' for x in h.history),'saw_calculation':any(x.get('tool_name')=='calculate' for x in h.history)}
    (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main(mode,root,repo,hcl_path=None,model_path=None):
    import sys
    root=Path(root);upstream=load_upstream(repo)
    sys.path.insert(0,str(repo))
    # The official scorer is invoked only after an episode; no benchmark gold is loaded.
    from medharness.scorers.calc import score_prediction
    model=None
    if mode=='real':
        sys.path.insert(0,str(hcl_path))
        from hcl.models import LocalTransformersModel
        backend=LocalTransformersModel(name='Qwen2.5-VL-7B-Instruct',path=model_path,torch_dtype='bfloat16',attn_implementation='sdpa',max_new_tokens=1024,answer_max_new_tokens=1024,temperature=0.,enable_thinking=False,local_files_only=True)
        model=LocalModel(backend,root)
    results=[]
    for case in CASES:
        result=asyncio.run(episode(upstream,root,case,model or ScriptedModel(case['script'])))
        gold={**case['gold'],'required_calculators':[r['name'] for r in case['gold']['calculator_answers']]}
        score=score_prediction(result['answer'] or {'calculators':[]},gold)
        result['score']=score
        result['passed']=result['termination']=='submitted' and score['utility']==1 and result['saw_facts'] and result['saw_calculation']
        results.append(result)
    receipt={'mode':mode,'cases':len(results),'passed':all(x['passed'] for x in results),'results':results,'model_calls':model.calls if model else 0,'scope':'synthetic engineering acceptance, not clinical benchmark accuracy or CL benefit','os_sandbox':False}
    (root/'RESULT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))
    return 0 if receipt['passed'] else 2

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['scripted','real'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--medicalharness',type=Path,required=True)
    parser.add_argument('--hcl',type=Path)
    parser.add_argument('--model')
    a=parser.parse_args()
    if a.mode=='real' and (not a.hcl or not a.model):parser.error('real mode requires hcl and model')
    a.output.mkdir(parents=True,exist_ok=True)
    raise SystemExit(main(a.mode,a.output,a.medicalharness,a.hcl,a.model))
