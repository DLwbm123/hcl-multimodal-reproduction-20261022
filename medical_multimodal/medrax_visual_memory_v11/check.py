"""CPU source/provenance, HCL restore, image-boundary and gate checks."""
import copy,json,os
from pathlib import Path
from episodic import ARMS,choose,context,gate

def main():
    root=Path(os.environ['R981_ROOT']);read=lambda n:json.loads((root/n).read_text())
    inputs=read('private/inputs.json');states=read('private/states.json');support=read('private/support_selection.json');gold=read('private/sealed_development_gold.json');records=states['historical_memory']['raw']
    assert len(inputs)==len(records)==len(gold)==24 and set(gold)=={x['id'] for x in inputs}
    assert all(set(x)=={'id','case_id','stage','role','question','images'} for x in inputs)
    train={r['internal_source']['task_id'] for r in records};assert len(train)==24 and not train&set(gold)
    from hcl.memory import ExperienceMemory,MemoryConfig
    import worker
    memory=ExperienceMemory(MemoryConfig(record_dir=str(root/'private/cpu_memory'),raw_capacity_per_task=24,abstract_capacity_per_task=2,skill_enabled=False,anchor_selection_mode='off'))
    for state in states.values():memory.restore(state);assert memory.snapshot()==state
    for item in inputs:
        r,m=choose(records,item['question']);assert support[item['id']]=={'selected':r['memory_id'],'mismatch':m['memory_id']}
        assert r['images']!=m['images'] and not set(r['images']+m['images'])&set(item['images'])
        assert context(r,False) in context(r,True)
    # A real existing tool boundary must reject support images before calling the underlying tool.
    from pydantic import BaseModel
    class Arguments(BaseModel):image_path:str
    class Tool:
        name='cpu_boundary_probe';description='CPU only';args_schema=Arguments
        def invoke(self,args):raise AssertionError('Historical image reached the tool')
    tool_state={'calls':0,'success':0,'images':set(inputs[0]['images'])};saved=worker.core.rt.record;worker.core.rt.record=lambda *args:None
    try:
        result=worker.core.rt.ObservedTool(Tool(),tool_state).invoke({'image_path':records[0]['images'][0]})
        assert result[1]['analysis_status']=='failed' and tool_state['success']==0
    finally:worker.core.rt.record=saved
    ids=['x','y'];labels={k:'A' for k in ids};by={}
    for arm in ARMS:
        for i in [1,2]:by[arm+'_r'+str(i)]={k:{'predicted':'A' if arm=='history_image' else 'B'} for k in ids}
    pairs=[{'H0_to_image':{'gain':2,'loss':0},'text_to_image':{'gain':1,'loss':0},'mismatch_to_image':{'gain':1,'loss':0}} for _ in [1,2]]
    assert gate(pairs,by,ids,labels,True)['pass'] and not gate(pairs,by,ids,labels,False)['pass']
    pairs[1]['text_to_image']['gain']=0;assert not gate(pairs,by,ids,labels,True)['pass']
    reserve=[r for r in json.loads((root.parent/'medrax_learning_qualification_v9/private/members.json').read_text()) if r['role']=='confirmation_reserved'];assert len(reserve)==96 and not {r['full_question_id'] for r in reserve}&(train|set(gold))
    result={'passed':True,'source_cases':24,'reused_development_cases':24,'confirmation_excluded':96,'fixed_retrieval_and_distinct_mismatch_checked':24,'raw_memory_restore_checked':True,'historical_image_rejected_by_real_current_case_tool_boundary':True,'repeated_gate_failure_checks':True,'real_model_calls':0}
    (root/'CPU_CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
