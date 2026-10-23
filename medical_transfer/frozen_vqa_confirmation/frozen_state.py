import os
import copy

def restore_frozen(pipe, state):
    assert state['stage']==3 and pipe.harness_version==0 and not pipe.checkpoint.has_state
    pipe.checkpoint.state={'memory_snapshot':copy.deepcopy(state['memory']),'artifacts':copy.deepcopy(state['artifacts'])}
    pipe.checkpoint.restore_pipeline(pipe)
    assert pipe.memory.snapshot()==state['memory']
    assert pipe.harness_version==state['artifacts']['harness_version']==1
    assert pipe.component_versions['memory_selector']['version']==1
    for k,v in pipe.component_versions.items():
        if k!='memory_selector':assert v['version']==0
    assert pipe.memory.summary()=={'raw':10,'abstract':2,'anchor':72,'skill':0}
    assert str(pipe.router_config.memory_selector_template_path)==state['artifacts']['paths']['memory_selector']
    return pipe
