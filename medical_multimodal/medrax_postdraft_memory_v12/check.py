"""CPU check of production turn ordering, state isolation and repair attribution."""
import copy,json,os
from pathlib import Path
from types import SimpleNamespace
from review_logic import ARMS,FINAL,review_text,decision


def main():
    root=Path(os.environ['R981_ROOT']);read=lambda n:json.loads((root/n).read_text());inputs=read('private/inputs.json');states=read('private/states.json');assert len(inputs)==6
    import worker
    from hcl.memory import ExperienceMemory,MemoryConfig
    from langchain_core.messages import AIMessage
    memory=ExperienceMemory(MemoryConfig(record_dir=str(root/'private/cpu_memory'),raw_capacity_per_task=24,abstract_capacity_per_task=2,skill_enabled=False,anchor_selection_mode='off'))
    for s in states.values():memory.restore(s);assert memory.snapshot()==s
    traces=[]
    class Model:
        def __init__(self):self.calls=0
    class Agent:
        def __init__(self,*args,**kwargs):self.messages=[];self.workflow=self;traces.append(self.messages)
        def invoke(self,value,config):
            msg=value['messages'][0];self.messages.append(msg.content)
            if len(self.messages)==1:answer='B. Frozen CPU draft'
            elif msg.content==FINAL:answer='B' if len(self.messages)==2 else 'A'
            else:answer='A. Frozen CPU review'
            return {'messages':[AIMessage(content=answer)]}
    scratch=root/'private/cpu_workflow';scratch.mkdir();(scratch/'outputs').mkdir();(scratch/'private').mkdir()
    saved=(worker.ROOT,worker.core.ROOT,worker.core.rt.SubscriptionModel,worker.core.EPISODES,worker.STATES)
    try:
        worker.ROOT=scratch;worker.core.ROOT=scratch;worker.core.rt.SubscriptionModel=Model;worker.STATES=states;worker.core.EPISODES=[]
        for arm in ARMS:worker.episode(inputs[0],arm,1,[],{},Agent,'CPU prompt',memory)
        assert len(traces[0])==2 and len(traces[1])==len(traces[2])==3
        assert traces[0][0]==traces[1][0]==traces[2][0]
        assert all(t[-1]==FINAL for t in traces)
        assert traces[1][1]==review_text('generic_review',states) and traces[2][1]==review_text('experience_review',states)
        assert all(r['initial_predicted']=='B' for r in worker.core.EPISODES)
        assert [r['review_entered'] for r in worker.core.EPISODES]==[False,True,True]
    finally:worker.ROOT,worker.core.ROOT,worker.core.rt.SubscriptionModel,worker.core.EPISODES,worker.STATES=saved
    ids=['x','y'];gold={k:'A' for k in ids};by={}
    for b in [1,2]:
        for a in ARMS:by[a+'_r'+str(b)]={k:{'predicted':'A' if a=='experience_review' else 'B','initial_predicted':'B','review_predicted':'A'} for k in ids}
    pairs=[{'H0_to_experience':{'gain':2,'loss':0},'generic_to_experience':{'gain':1,'loss':0}} for _ in [1,2]]
    assert decision(pairs,by,ids,gold,True)['pass'] and not decision(pairs,by,ids,gold,False)['pass']
    by['experience_review_r1']['x']['initial_predicted']='A';assert not decision(pairs,by,ids,gold,True)['pass']
    result={'passed':True,'production_episode_workflow_checked':True,'initial_input_identical_across_all_three_arms':True,'only_post_draft_review_receives_procedures':True,'final_author_turn_preserved':True,'memory_restore_and_immutability_checked':True,'gate_rejects_preexisting_initial_correctness_as_repair':True,'incomplete_gate_rejected':True,'real_model_calls':0};(root/'CPU_CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
