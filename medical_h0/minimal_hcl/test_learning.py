import asyncio,json,tempfile,unittest,sys
from pathlib import Path
from learning import optimizer,choose,commit,validate
from data import build
from experiment import episode
from adapter import load_upstream
from qwen_transport import LocalModel
from test_transport import StubProtocol
from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import NousFnCallPrompt
from langchain_core.messages import AIMessage

def rs(bits):return [dict(case_id=str(i),success=bool(x),utility=float(x),submitted=True) for i,x in enumerate(bits)]
class Script:
    def __init__(self,actions):self.actions=list(actions);self.calls=0
    def bind_tools(self,s):return self
    async def ainvoke(self,messages):
        self.calls+=1;n,a=self.actions.pop(0)
        return AIMessage(content='',tool_calls=[{'name':n,'args':a,'id':str(self.calls)}])
class Tests(unittest.TestCase):
    def test_gate_commit_and_loss_not_offset(self):
        with tempfile.TemporaryDirectory() as d:
            o=optimizer(d);c={'candidate_id':'x','artifact_content':{'sections':['Check task requirements.']},'validation_errors':[]}
            before=rs([1,0,0,0]);hist=rs([1,0,1,0]);ev={'x':{'current':rs([1,1,0,0]),'anchors':rs([0,1,1,0])}}
            self.assertIsNone(choose(o,[c],ev,before,hist))
            ev['x']['anchors']=hist
            self.assertEqual(choose(o,[c],ev,before,hist),c)
            path=Path(d)/'state.json';state={'version':0,'sections':[]}
            self.assertEqual(commit(path,state,None,1),state)
            new=commit(path,state,c,1);self.assertEqual(new['version'],1)
            self.assertEqual(commit(path,new,None,2),new)
    def test_candidate_raw_official_protocol(self):
        from learning import CANDIDATE_SCHEMA
        p=StubProtocol('<tool_call>\n'+json.dumps({'name':'propose_workflow','arguments':{'candidate_name':'a','rationale':'learned','sections':['Check requested outputs.'],'artifact_diff_summary':'workflow'}})+'\n</tool_call>')
        p.fncall_prompt=NousFnCallPrompt();p.generate_cfg['stop']=[]
        raw=p.chat([{'role':'user','content':'Propose'}],functions=[CANDIDATE_SCHEMA['function']],stream=False)
        bridge=LocalModel(None,'.').bind_tools([CANDIDATE_SCHEMA]);result=bridge.decode(raw)
        self.assertEqual(result.tool_calls[0]['args']['sections'],['Check requested outputs.'])
        bad=bridge.decode([{'function_call':{'name':'propose_workflow','arguments':'{"sections": []}'}}]);self.assertTrue(bad.invalid_tool_calls)
        self.assertTrue(validate(['Answer is 12.']));self.assertTrue(validate(['import os']))
    def test_disjoint_roles(self):
        cases=build();self.assertEqual(len(cases),36)
        self.assertEqual(len({json.dumps((c['prompt'],c['rows'])) for c in cases}),36)
    def test_all_oracles_and_workflow_isolation(self):
        repo=Path(sys.argv_repo);sys.path.insert(0,str(repo))
        from medharness.scorers.calc import score_prediction
        u=load_upstream(repo)
        with tempfile.TemporaryDirectory() as d:
            for c in build():
                visible={k:c[k] for k in ('id','prompt','rows')}
                r=asyncio.run(episode(u,d,visible,Script(c['script']),[]))
                self.assertEqual(score_prediction(r['answer'],c['gold'])['utility'],1,c['id'])
                self.assertEqual(r['termination'],'submitted')
    def test_unsubmitted_is_not_normal(self):
        class Final(Script):
            async def ainvoke(self,messages):self.calls+=1;return AIMessage(content='Done')
        u=load_upstream(Path(sys.argv_repo))
        with tempfile.TemporaryDirectory() as d:
            r=asyncio.run(episode(u,d,build()[0],Final([]),[]))
            self.assertEqual(r['termination'],'unsubmitted_final')
if __name__=='__main__':
    sys.argv_repo=sys.argv.pop(1);unittest.main()
