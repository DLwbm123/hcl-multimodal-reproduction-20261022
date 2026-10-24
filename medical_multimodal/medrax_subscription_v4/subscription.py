"""Local ChatGPT subscription transport. Authentication never leaves the Mac."""
import base64
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

BINARY = '/Applications/ChatGPT.app/Contents/Resources/codex-cli/CodexCLI.app/Contents/MacOS/codex'
SCHEMA = {'type': 'object', 'properties': {'content': {'type': 'string'}, 'tool_calls': {'type': 'array', 'items': {'type': 'object', 'properties': {'name': {'type': 'string'}, 'arguments_json': {'type': 'string'}}, 'required': ['name', 'arguments_json'], 'additionalProperties': False}}}, 'required': ['content', 'tool_calls'], 'additionalProperties': False}

def rpc(action, payload=None):
    p = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'pro5000', '/data/bmw/envs/p0/bin/python', '/data/bmw/workers/r984.py'], input=json.dumps({'action': action, 'payload': payload}), text=True, capture_output=True, timeout=60)
    if p.returncode: raise RuntimeError('RPC failed: '+p.stderr[-500:])
    return json.loads(p.stdout)

def execute(job, local_root):
    budget = json.loads((local_root/'budget.json').read_text())
    counter = local_root/'private/call_count.json'
    used = json.loads(counter.read_text()) if counter.exists() else 1  # Initial verified CLI smoke.
    if used >= budget['max_subscription_calls'] or time.time() >= budget['hard_deadline_epoch']:
        raise RuntimeError('Subscription budget/deadline exhausted')
    counter.write_text(json.dumps(used+1))
    with tempfile.TemporaryDirectory(prefix='r983-',dir='/tmp') as temp:
        p = Path(temp)
        home=p/'auth'; home.mkdir(mode=0o700)
        (home/'auth.json').symlink_to(Path.home()/'.codex/auth.json')
        (p/'instructions.txt').write_text(job['system'])
        (p/'schema.json').write_text(json.dumps(SCHEMA))
        (home/'config.toml').write_text('''model = "gpt-6.1-sol"
model_reasoning_effort = "high"
approval_policy = "never"
sandbox_mode = "read-only"
web_search = "disabled"
project_doc_max_bytes = 0
model_instructions_file = "'''+str(p/'instructions.txt')+'''"
[features]
memories = false
apps = false
plugins = false
shell_tool = false
unified_exec = false
view_image = false
image_generation = false
multi_agent = false
hooks = false
shell_snapshot = false
skill_search = false
skip_host_skill_discovery = true
''')
        user_home=Path.home()
        (p/'policy.sb').write_text(f'''(version 1)
(allow default)
(deny file-read* (subpath "{user_home}/Documents") (subpath "{user_home}/Downloads") (subpath "{user_home}/Desktop") (subpath "{user_home}/.ssh") (subpath "{user_home}/.codex/memories") (subpath "{user_home}/.codex/memories_v2") (subpath "{user_home}/.codex/instructions"))
(deny process-exec)
(allow process-exec (literal "{BINARY}"))
''')
        cmd=['/usr/bin/sandbox-exec','-f',str(p/'policy.sb'),BINARY,'exec','--ephemeral','--ignore-rules','--skip-git-repo-check','-C',str(p),'--json','--output-schema',str(p/'schema.json'),'-o',str(p/'final.json')]
        for i, encoded in enumerate(job.get('images',[])):
            image=p/f'i{i}.png'; image.write_bytes(base64.b64decode(encoded)); cmd+=['--image',str(image)]
        cmd+=['-']
        env=os.environ.copy(); env['CODEX_HOME']=str(home)
        for key in ['OPENAI_API_KEY','CODEX_API_KEY','OPENAI_BASE_URL']: env.pop(key,None)
        prompt='Continue the provided conversation as its assistant. The images attached correspond in order to the image entries in the conversation. Available medical tools are external functions executed by the caller. Return your next assistant message in the output schema; tool_calls contains chosen function names and JSON-encoded arguments, or is empty for a final response. Do not simulate tool results.\n'+json.dumps({'tools':job['tools'],'messages':job['messages']},ensure_ascii=False)
        started=time.time(); proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=env,start_new_session=True)
        (local_root/'private/active_cli.json').write_text(json.dumps({'pid':proc.pid,'started':started,'image_count':len(job.get('images',[]))}))
        timeout=min(1200,budget['hard_deadline_epoch']-time.time())
        try: stdout,stderr=proc.communicate(prompt,timeout=max(1,timeout))
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try: stdout,stderr=proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL); stdout,stderr=proc.communicate()
            return {'status':'timeout','seconds':time.time()-started,'stderr':stderr[-1000:]}
        events=[json.loads(x) for x in stdout.splitlines() if x.startswith('{')]
        forbidden=[e for e in events if e.get('item',{}).get('type') in {'command_execution','web_search','mcp_tool_call'}]
        result={'status':'ok' if proc.returncode==0 and not forbidden else 'cli_failed','exit_code':proc.returncode,'events':events,'stderr':stderr[-4000:],'seconds':time.time()-started,'image_count':len(job.get('images',[])),'model':budget['model'],'reasoning':budget['reasoning']}
        if result['status']=='ok':
            value=json.loads((p/'final.json').read_text())
            assert set(value)=={'content','tool_calls'} and isinstance(value['content'],str) and isinstance(value['tool_calls'],list)
            for call in value['tool_calls']:
                assert set(call)=={'name','arguments_json'} and isinstance(call['name'],str)
                assert isinstance(json.loads(call['arguments_json']),dict)
            result['output']=value
        return result

def refresh_urls(root):
    import urllib.request, urllib.error
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self,req,fp,code,msg,headers,newurl): return None
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({'https':'http://127.0.0.1:7897'}),NoRedirect())
    token=os.environ.get('HF_TOKEN') or (Path.home()/'.cache/huggingface/token').read_text().strip()
    updates=[]
    for directory in ['maira-2','llava-med-mistral']:
        meta=json.loads((root/'private'/(directory+'_metadata.json')).read_text())
        for file in meta['siblings']:
            name=file['rfilename']
            if '/' in name or not name.endswith(('.json','.py','.model','.safetensors')) or not file.get('lfs'):continue
            url='https://huggingface.co/'+meta['id']+'/resolve/'+meta['sha']+'/'+name+'?download=true&refresh='+str(int(time.time()))
            req=urllib.request.Request(url,headers={'Authorization':'Bearer '+token},method='HEAD')
            try:
                with opener.open(req,timeout=60) as response:resolved=response.url
            except urllib.error.HTTPError as exc:
                if exc.code not in (301,302,303,307,308):raise RuntimeError('URL refresh HTTP '+str(exc.code)) from None
                resolved=exc.headers['Location']
            updates.append({'directory':directory,'name':name,'url':resolved})
    return rpc('refresh_urls',updates)

def main():
    root=Path(os.environ['R983_LOCAL'])
    deadline=json.loads((root/'budget.json').read_text())['hard_deadline_epoch']
    refresh_after=time.time()+1200
    while time.time()<deadline:
        try:
            if time.time()>=refresh_after:
                refresh_urls(root);refresh_after=time.time()+1200
            job=rpc('pull')
            if job['state']=='finished': return
            if job['state']!='job': time.sleep(3); continue
            path=root/'private'/('response_'+job['key']+'.json')
            if path.exists(): response=json.loads(path.read_text())
            else:
                response=execute(job['job'],root); path.write_text(json.dumps(response))
            rpc('respond',{'key':job['key'],'response':response})
            print(json.dumps({'key':job['key'],'status':response['status'],'seconds':response.get('seconds')}),flush=True)
        except Exception as exc:
            print(json.dumps({'error':repr(exc),'time':time.time()}),flush=True)
            time.sleep(10)

if __name__=='__main__': main()
