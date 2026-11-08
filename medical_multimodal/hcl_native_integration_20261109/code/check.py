"""Synthetic safety checks; these are not Agent or learning results."""
import json
from copy import deepcopy
from pathlib import Path
import tempfile
from state_store import StateStore, gate
from preflight import check_inputs


def row(key, answer, status='completed'):
    return {'id':key,'predicted':answer,'status':status}


def main(root):
    preflight=check_inputs(root,json.loads((root/'private/inputs.json').read_text()),False)
    current=[row('x','B'),row('y','A')]
    candidate=[row('x','A'),row('y','A')]
    labels={'x':'A','y':'A'}
    assert gate(current,candidate,labels,['x','y'])['accept']
    # A recovered anchor cannot compensate for a previously correct one lost.
    d=gate(current,candidate,labels,['x','y'],[row('a','A'),row('b','B')],
           [row('a','B'),row('b','A')],{'a':'A','b':'A'},['a','b'])
    assert not d['accept'] and d['anchor_losses']==1 and d['anchor_recovered']==1
    assert not gate(current,candidate[:1],labels,['x','y'])['accept']
    assert not gate(current,[row('x',None,'failed'),row('y','A')],labels,['x','y'])['accept']
    with tempfile.TemporaryDirectory() as tmp:
        store=StateStore(tmp);original={'memory':{'raw':[]},'version':0}
        store.deploy('S0',original,None,{'initial':True})
        rejected=deepcopy(original);rejected['memory']['raw'].append({'synthetic':True})
        store.save('uncommitted',rejected)
        # Reopening after a candidate write or an interrupted pointer temp restores S0.
        (Path(tmp)/'DEPLOYED.tmp').write_text('{')
        reopened=StateStore(tmp);assert reopened.load()==original
        reopened.deploy('SA',original,'S0',{'accept':False})
        assert StateStore(tmp).load()==original
        accepted=deepcopy(rejected);accepted['version']=1
        reopened.deploy('SAB',accepted,'SA',{'synthetic_test_accept':True})
        assert StateStore(tmp).load()==accepted
    from hcl.task_interface import TaskInterface,TaskInterfaceConfig
    chunk=TaskInterface(TaskInterfaceConfig(mode='rule')).build_taskinterfacechunk(
        {'task_id':'synthetic','question':'Choose an option based on attached evidence.',
         'answer':'HIDDEN_REFERENCE_SENTINEL','split':'report','task_name':'HIDDEN_STAGE_SENTINEL'},0)
    visible=json.dumps(TaskInterface.to_model_visible(chunk))
    assert 'HIDDEN_REFERENCE_SENTINEL' not in visible and 'HIDDEN_STAGE_SENTINEL' not in visible
    from hcl.memory import ExperienceMemory,MemoryConfig
    from hcl.router import Router,RouterConfig
    with tempfile.TemporaryDirectory() as tmp:
        memory=ExperienceMemory(MemoryConfig(record_dir=tmp,skill_enabled=False,anchor_selection_mode='off'))
        example={'task_id':'source','question':'Check evidence before selecting an option.',
                 'answer':'A','split':'train'}
        memory.record_training_interaction('source',example,{'task_id':'source','answer':'B'})
        frozen=deepcopy(memory.snapshot())
        memory_id=memory.raw.records[0]['memory_id']
        class Selector:
            def generate(self,prompt,*,state):
                assert 'HIDDEN_REFERENCE_SENTINEL' not in prompt and 'HIDDEN_STAGE_SENTINEL' not in prompt
                return json.dumps({'selected':[{'memory_id':memory_id,'role':'failure_avoidance',
                    'reason':'Synthetic selector contract test','risk':'low'}],
                    'discarded_summary':'none','selection_confidence':'high'})
        router=Router(Selector(),RouterConfig(use_llm_workflow_selector=False,
                       use_skill_selector=False,use_tool_selector=False),memory=memory)
        workflow=router.plan_workflow(chunk,selected_skills=[])
        selected,trace=router.select_memory(chunk,workflow)
        assert len(selected)==1 and selected[0]['memory_id']==memory_id
        assert memory.snapshot()==frozen
    return {'passed':True,'scope':'synthetic CPU tests only',
            'checks':['gain gate','anchor loss not offset','missing denominator rejected',
                      'failed output rejected','uncommitted/interrupted pointer recovery',
                      'rejection preserves deployed state','accepted snapshot reload',
                      'current labels and task identity excluded from visible interface',
                      'real Router schema/materialization and read-only memory with synthetic selector'],
            'preflight':preflight}


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    result=main(root)
    (root/'CPU_CHECK.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
