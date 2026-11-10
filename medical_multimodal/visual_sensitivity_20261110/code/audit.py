"""Fixed image-conditioning controls; does not access task targets."""
import ast,json,re,signal,time,traceback
from pathlib import Path
from PIL import Image
import base_worker as base
from qualified_tool import install_shared_guard
ROOT,PLAN=base.ROOT,base.PLAN
ROWS=[]

def normalized(value):return ' '.join(str(value).lower().split())
def similarity(a,b):
    x=set(re.findall(r'\w+',normalized(a)));y=set(re.findall(r'\w+',normalized(b)))
    return len(x&y)/len(x|y) if x|y else 1.0

def summarize(rows):
    lookup={(x['audit_id'],x['tool'],x['condition']):x for x in rows};out={}
    for tool in PLAN['tools']:
        records=[]
        for i in range(6):
            key=f'u{i:02d}';cells={c:lookup.get((key,tool,c)) for c in PLAN['conditions']}
            if not all(x and x['status']=='completed' for x in cells.values()):continue
            real=cells['real']['response'];records.append({c:{'same_normalized_text':normalized(real)==normalized(cells[c]['response']),'token_jaccard':similarity(real,cells[c]['response'])} for c in PLAN['conditions'][1:]})
        out[tool]={'planned_sources':6,'complete_control_sets':len(records),'comparisons':{c:{'text_changed_sources':sum(not x[c]['same_normalized_text'] for x in records),'mean_token_jaccard':sum(x[c]['token_jaccard'] for x in records)/len(records) if records else None} for c in PLAN['conditions'][1:]}}
    return {'planned_calls':48,'recorded_calls':len(rows),'successful_calls':sum(x['status']=='completed' for x in rows),'coverage_complete':len(rows)==48,'by_tool':out,'clinical_accuracy_measured':False,'method_effectiveness_measured':False,'labels_accessed':False}

def main():
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('Independent supervisor stop')))
    items=json.loads((ROOT/'private/inputs.json').read_text());assert len(items)==6
    base.save('STARTED.json',{'time':time.time(),'original_start':base.BUDGET['started_epoch'],'planned_calls':48})
    fatal=None
    try:
        wrappers,state,_,_=base.load_four();install_shared_guard(wrappers,state,base.record);tools={t.name:t for t in wrappers}
        base.save('TOOLS_LOADED.json',{'time':time.time(),'audit_tools':PLAN['tools']})
        for i,item in enumerate(items):
            real=item['image'];other=items[(i+1)%6]['image']
            with Image.open(real) as im:size=im.size
            blank=ROOT/'private'/f'black_{size[0]}x{size[1]}.png'
            if not blank.exists():Image.new('RGB',size,(0,0,0)).save(blank)
            paths={'real':real,'repeat_real':real,'cyclic_other_image':other,'uniform_black':str(blank)}
            for tool_name in PLAN['tools']:
                for condition,path in paths.items():
                    base.guard();base.CONTEXT.update(phase='image_sensitivity',audit_id=item['audit_id'],tool=tool_name,condition=condition)
                    state.update(calls=0,success=0,images={str(Path(path).resolve())})
                    args={'image_paths':[path],'prompt':PLAN['expert_prompt'],'max_new_tokens':256} if tool_name=='chest_xray_expert' else {'image_path':path}
                    start=time.time();result=tools[tool_name].invoke(args)
                    status=result[1].get('analysis_status','unknown') if isinstance(result,tuple) else 'unknown'
                    response=result[0];response=response.get('response',response) if isinstance(response,dict) else response
                    ROWS.append({'audit_id':item['audit_id'],'tool':tool_name,'condition':condition,'status':status,'response':response,'seconds':time.time()-start})
                    base.save('private/results.json',ROWS);base.save('PROGRESS.json',{'time':time.time(),'recorded_calls':len(ROWS),'planned_calls':48})
                    base.guard()
    except BaseException as exc:
        fatal=repr(exc);base.save('FAILED.json',{'error':fatal,'traceback':traceback.format_exc()})
    finally:
        report=summarize(ROWS);report.update(fatal=fatal,elapsed_seconds=time.time()-base.BUDGET['started_epoch']);base.save('REPORT.json',report);base.save('FINISHED.json',report)
    return 2 if fatal else 0

if __name__=='__main__':raise SystemExit(main())
