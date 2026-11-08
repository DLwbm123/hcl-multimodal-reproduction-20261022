"""Author six-tool MedRAX graph, with a local subscription controller and no HCL."""
import base64
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid

ROOT=Path(os.environ['R981_ROOT'])
# Reuse the host's matching Python headers for Triton's small CUDA helper build.
os.environ['CPATH']='/data/bmw/envs/n3-headers/usr/include/python3.12:/data/bmw/envs/n3-headers/usr/include'
os.environ['TMPDIR']='/data/bmw/workers/t981'
os.environ['TRITON_CACHE_DIR']='/data/bmw/workers/t981/triton'

PLAN=json.loads((ROOT/'code/protocol.json').read_text())
BUDGET=json.loads((ROOT/'budget.json').read_text())
CONTEXT={'phase':'waiting_assets','episode':None}

def save(name,value):
    p=ROOT/name; t=p.with_suffix('.tmp'); t.write_text(json.dumps(value,indent=2,default=str));t.replace(p)

def record(kind,value):
    with (ROOT/'private/events.jsonl').open('a') as f:
        f.write(json.dumps({'time':time.time(),**CONTEXT,'kind':kind,**value},default=str)+'\n')

def guard():
    if time.time()>=BUDGET['hard_deadline_epoch']: raise RuntimeError('Original deadline reached')
    if (ROOT/'STOPPED.json').exists(): raise RuntimeError('Storage guard stopped')

def watchdog():
    while True:
        used=int(subprocess.check_output(['du','-sb',str(ROOT)]).split()[0])
        save('RUN_GUARD.json',{'time':time.time(),'bytes':used,'deadline':BUDGET['hard_deadline_epoch']})
        if used>=BUDGET['stop_new_bytes'] or time.time()>=BUDGET['hard_deadline_epoch']:
            save('STOPPED.json',{'time':time.time(),'bytes':used,'phase':CONTEXT})
            os.killpg(os.getpgrp(),signal.SIGTERM)
        time.sleep(15)

class SubscriptionModel:
    def __init__(self): self.calls=0; self.tools=[]; self.deadline=0
    def bind_tools(self,tools):
        self.tools=[{'name':t.name,'description':t.description,'parameters':t.args_schema.model_json_schema()} for t in tools]
        return self
    def invoke(self,messages):
        from langchain_core.messages import AIMessage
        from PIL import Image
        guard();self.calls+=1
        if self.calls>PLAN['limits']['per_case_controller_calls'] or time.time()>self.deadline: raise RuntimeError('Episode budget exhausted')
        job={'system':'','tools':self.tools,'messages':[],'images':[]}
        for m in messages:
            if m.type=='system':job['system']=m.content;continue
            value={'role':{'human':'user','ai':'assistant','tool':'tool'}[m.type],'content':m.content}
            if isinstance(m.content,list):
                content=[]
                for part in m.content:
                    if part['type']=='image_url':
                        path=Path(part['image_url']['url'])
                        with Image.open(path) as im:
                            if getattr(im,'n_frames',1)!=1: raise ValueError('Animated input requires separate protocol decision')
                            out=io.BytesIO();im.convert('RGB').save(out,format='PNG')
                        index=len(job['images']);job['images'].append(base64.b64encode(out.getvalue()).decode())
                        content.append({'type':'image','index':index,'path':str(path)})
                    else: content.append(part)
                value['content']=content
            if getattr(m,'tool_calls',None):value['tool_calls']=m.tool_calls
            if m.type=='tool':value.update(tool_call_id=m.tool_call_id,name=m.name)
            job['messages'].append(value)
        key=uuid.uuid4().hex; queue=ROOT/'private/queue'
        temp=queue/(key+'.tmp');temp.write_text(json.dumps(job));temp.replace(queue/(key+'.request.json'))
        record('controller_started',{'key':key,'images':len(job['images']),'tool_results':sum(m['role']=='tool' for m in job['messages'])})
        response=queue/(key+'.response.json'); wait_until=min(self.deadline,time.time()+PLAN['limits']['request_timeout_seconds']+120)
        while not response.exists():
            guard()
            if time.time()>wait_until:raise TimeoutError('Local subscription response timed out')
            time.sleep(2)
        result=json.loads(response.read_text())
        record('controller_finished',{'key':key,'status':result['status'],'seconds':result.get('seconds'),'image_count':result.get('image_count'),'usage':[e['usage'] for e in result.get('events',[]) if e.get('type')=='turn.completed']})
        if result['status']!='ok':raise RuntimeError('Subscription request failed: '+result['status'])
        v=result['output']
        calls=[{'id':uuid.uuid4().hex,'name':c['name'],'args':json.loads(c['arguments_json']),'type':'tool_call'} for c in v['tool_calls']]
        return AIMessage(content=v['content'],tool_calls=calls)

class ObservedTool:
    def __init__(self,tool,state):
        self.tool,self.state=tool,state
        self.name,self.description,self.args_schema=tool.name,tool.description,tool.args_schema
    def invoke(self,args):
        guard();self.state['calls']+=1
        if self.state['calls']>PLAN['limits']['per_case_tool_calls']:raise RuntimeError('Tool-call budget exhausted')
        start=time.time()
        try:
            parsed=self.args_schema.model_validate(args).model_dump()
            paths=parsed.get('image_paths',[parsed.get('image_path')])
            if any(p is not None and str(Path(p).resolve()) not in self.state['images'] for p in paths):raise ValueError('Image outside current case')
        except Exception as exc:
            result=({'error':str(exc)},{'analysis_status':'failed'})
        else:
            record('tool_started',{'name':self.name,'args':args})
            result=self.tool.invoke(args)
        status=result[1].get('analysis_status','unknown') if isinstance(result,tuple) else 'unknown'
        record('tool_finished',{'name':self.name,'status':status,'seconds':time.time()-start,'result':str(result)})
        if status in {'completed','completed_no_finding'}:self.state['success']+=1
        return result

def load_six():
    from native_runtime import load_native,load_module
    import torch
    tools,_,agent,prompt=load_native(ROOT,record,guard)
    by={t.name:t.tool for t in tools}
    td=ROOT/'upstream/medrax/tools'
    grounding=load_module(td/'grounding.py','subscription_grounding').XRayPhraseGroundingTool(model_path=str(ROOT/'models/maira-2'),cache_dir=str(ROOT/'cache'),temp_dir=str(ROOT/'outputs'),device='cuda',load_in_8bit=True)
    # A separate small config points the unmodified builder to the existing CLIP asset.
    original=ROOT/'models/llava-med-mistral'; runtime=ROOT/'models/llava-med-mistral-runtime';runtime.mkdir(exist_ok=True)
    for source in original.iterdir():
        dest=runtime/source.name
        if source.name=='config.json':
            cfg=json.loads(source.read_text());cfg['mm_vision_tower']='/data/bmw/HCL/models/clip-vit-large-patch14-336';dest.write_text(json.dumps(cfg))
        elif not dest.exists():dest.symlink_to(source)
    sys.path.insert(0,str(ROOT/'upstream'))
    llava=load_module(td/'llava_med.py','subscription_llava').LlavaMedTool(model_path=str(runtime),cache_dir=str(ROOT/'cache'),device='cuda',load_in_8bit=True)
    # Keep the author's six-tool order, which also preserves schema presentation order.
    native=[by['chest_xray_report_generator'],by['chest_xray_classifier'],by['chest_xray_segmentation'],grounding,by['chest_xray_expert'],llava]
    state={'calls':0,'success':0,'images':set()}
    return [ObservedTool(t,state) for t in native],state,agent,prompt

def main():
    threading.Thread(target=watchdog,daemon=True).start()
    save('STARTED.json',{'time':time.time(),'original_start':BUDGET['started_epoch'],'deadline':BUDGET['hard_deadline_epoch'],'hcl_updates':0})
    while not (ROOT/'ASSETS_READY.json').exists():
        guard()
        if (ROOT/'DOWNLOAD_EXIT.json').exists():raise RuntimeError('Asset preparation failed; original clock retained')
        time.sleep(15)
    free=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    available={int(row.split(',')[0]):int(row.split(',')[1]) for row in free.strip().splitlines()}
    gpu=int(os.environ['CUDA_VISIBLE_DEVICES']);save('GPU_PREFLIGHT.json',{'time':time.time(),'gpu':gpu,'free_mib':available[gpu]})
    if available[gpu]<45000:raise RuntimeError('GPU headroom below 45000 MiB; no existing process touched')
    CONTEXT['phase']='loading_six_tools'
    tools,state,Agent,prompt=load_six()
    save('TOOLS_LOADED.json',{'time':time.time(),'names':[t.name for t in tools]})
    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.memory import MemorySaver
    panel=json.loads((ROOT/'private/panel.json').read_text());images_root=ROOT/'data/chestagentbench/images'
    image=str((images_root/panel[0]['images'][0]).resolve());state['images']={image}
    CONTEXT['phase']='six_tool_preflight'; preflight=[]
    for tool in tools:
        if tool.name=='chest_xray_expert':args={'image_paths':[image],'prompt':'Describe the chest X-ray findings.'}
        elif tool.name=='xray_phrase_grounding':args={'image_path':image,'phrase':'Pleural effusion'}
        elif tool.name=='llava_med_qa':args={'image_path':image,'question':'Describe the chest X-ray findings.'}
        else:args={'image_path':image}
        result=tool.invoke(args); status=result[1].get('analysis_status') if isinstance(result,tuple) else None
        preflight.append({'tool':tool.name,'status':status,'success':status in {'completed','completed_no_finding'}})
        save('TOOL_PREFLIGHT.json',preflight)
    if not all(r['success'] for r in preflight):raise RuntimeError('Six-tool preflight failed; qualification panel not started')
    # Only the allowlisted question/image fields are sent to controller or tool execution.
    rows={r['full_question_id']:{k:r[k] for k in ['question','images']} for r in map(json.loads,(ROOT/'data/chestagentbench/metadata.jsonl').read_text().splitlines())}
    results=[]
    for index,item in enumerate(panel):
        guard();CONTEXT.update(phase='native_panel',episode=index)
        paths=[str((images_root/name).resolve()) for name in item['images']]
        state.update(calls=0,success=0,images=set(paths))
        model=SubscriptionModel();model.deadline=min(time.time()+PLAN['limits']['per_case_seconds'],BUDGET['hard_deadline_epoch'])
        logdir=ROOT/'outputs'/str(index);logdir.mkdir(exist_ok=True)
        agent=Agent(model,tools,system_prompt=prompt,checkpointer=MemorySaver(),log_dir=str(logdir))
        config={'configurable':{'thread_id':'1'},'recursion_limit':2*PLAN['limits']['per_case_controller_calls']+2}
        figure_prompt=''.join(Path(p).name+' located at '+p+'\n' for p in paths)
        text=('Answer this question correctly using chain of thought reasoning and carefully evaluating choices. Solve using our own vision and reasoning and thenuse tools to complement your reasoning. Trust your own judgement over any tools.\n'+rows[item['full_question_id']]['question']+'\n'+figure_prompt)
        content=[{'type':'text','text':text}]+[{'type':'image_url','image_url':{'url':p}} for p in paths]
        result={'index':index,'id':item['full_question_id'],'status':'failed','predicted':None};start=time.time()
        try:
            output=agent.workflow.invoke({'messages':[HumanMessage(content=content)]},config)
            result['initial_final']=output['messages'][-1].content
            output=agent.workflow.invoke({'messages':[HumanMessage(content='If you had to choose the best option, only respond with the letter of choice (only one of A, B, C, D, E, F)')]},config)
            final=output['messages'][-1].content.strip()
            result.update(status='completed',final=final,predicted=final if final in list('ABCDEF') else None)
        except Exception as exc:
            result['error']=repr(exc);record('episode_error',{'traceback':traceback.format_exc()})
        result.update(seconds=time.time()-start,controller_calls=model.calls,tool_calls=state['calls'],successful_tools=state['success'])
        results.append(result);save('private/results.json',results)
    # Gold is introduced only after every frozen case has finished.
    gold={r['full_question_id']:str(r['answer']).strip()[0] for r in map(json.loads,(ROOT/'data/chestagentbench/metadata.jsonl').read_text().splitlines())}
    for r in results:r['correct']=r['predicted']==gold[r['id']]
    save('private/results.json',results)
    summary={'time':time.time(),'denominator':len(panel),'completed':sum(r['status']=='completed' for r in results),'valid_final':sum(r['predicted'] is not None for r in results),'correct':sum(r['correct'] for r in results),'episodes_with_successful_tool':sum(r['successful_tools']>0 for r in results),'controller_calls':sum(r['controller_calls'] for r in results),'tool_calls':sum(r['tool_calls'] for r in results),'preflight':preflight,'hcl_updates':0,'scientific_efficacy_claim':False}
    summary['operational_qualified']=summary['completed']==len(panel) and summary['valid_final']==len(panel) and summary['episodes_with_successful_tool']>0
    save('FINISHED.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        save('FAILED.json',{'time':time.time(),'error':repr(exc),'phase':CONTEXT,'traceback':traceback.format_exc()});raise
