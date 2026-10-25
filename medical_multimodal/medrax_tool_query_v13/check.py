"""CPU schema, real tool boundary and information-path tests, with no model call."""
import json,os
from pathlib import Path
from pydantic import BaseModel
from query_logic import FIXED_QUERY,validate,query,prefix_text,decision,stem


def main():
    root=Path(os.environ['R981_ROOT']);inputs=json.loads((root/'private/inputs.json').read_text());assert len(inputs)==6 and all(2<=len(x['images'])<=4 for x in inputs)
    policy={'admit':True,'tool_name':'chest_xray_expert','query_template':FIXED_QUERY+' Check the evidence before drawing an interpretation.','source_indices':[0,1,2,3],'rationale':'CPU-only fixture'};validate(policy)
    bad=dict(policy,query_template='{unsafe}'+policy['query_template'])
    try:validate(bad)
    except AssertionError:pass
    else:raise AssertionError('Unknown placeholder admitted')
    import worker
    class Args(BaseModel):image_paths:list[str];prompt:str
    seen=[]
    class Tool:
        name='chest_xray_expert';description='CPU fixture';args_schema=Args
        def invoke(self,args):seen.append(args);return {'response':'CPU observed finding'},{'analysis_status':'completed'}
    item=inputs[0];state={'calls':0,'success':0,'images':set(item['images'])};tool=worker.core.rt.ObservedTool(Tool(),state);saved=worker.core.rt.record;worker.core.rt.record=lambda *a:None
    try:
        rows=worker.execute_prefix(item,policy,[tool]);assert state['calls']==len(item['images'])==len(rows)==len(seen)
        assert all(a['image_paths']==[item['images'][i]] and a['prompt']==query(policy,item['question']) for i,a in enumerate(seen))
        assert stem('Visible question\nA. first choice\nB. second choice')=='Visible question'
        assert all(set(x)=={'current_image_index','tool','status','observation'} for x in json.loads(prefix_text(rows).split('\n',2)[-1]))
        rendered=prefix_text(rows);assert policy['query_template'] not in rendered and policy['rationale'] not in rendered and '{question_stem}' not in rendered
        assert worker.execute_prefix(item,None,[tool])==[]
    finally:worker.core.rt.record=saved
    ids=['x','y'];gold={k:'A' for k in ids};by={}
    for b in [1,2]:
        by['H0_static_r'+str(b)]={k:{'predicted':'B'} for k in ids};by['fixed_query_r'+str(b)]={k:{'predicted':'B','prefix_text':'same'} for k in ids};by['learned_query_r'+str(b)]={k:{'predicted':'A','prefix_text':'different'} for k in ids}
    pairs=[{'H0_to_learned':{'gain':2,'loss':0},'fixed_to_learned':{'gain':1,'loss':0}} for _ in [1,2]];assert decision(pairs,by,ids,gold,True)['pass'] and not decision(pairs,by,ids,gold,False)['pass']
    by['learned_query_r1']['x']['prefix_text']='same';assert not decision(pairs,by,ids,gold,True)['pass']
    result={'passed':True,'restricted_program_schema_and_placeholder_checked':True,'production_prefix_invokes_one_real_wrapped_tool_per_current_image':True,'queries_use_current_stem_without_answer_options':True,'controller_observations_exclude_program_query_and_source_feedback':True,'H0_has_no_prefix_calls':True,'identical_observation_gain_not_counted_as_source_intervention':True,'incomplete_gate_rejected':True,'real_model_calls':0};(root/'CPU_CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
