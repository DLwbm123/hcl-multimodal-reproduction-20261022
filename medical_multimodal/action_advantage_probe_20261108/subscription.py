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

def normalize_image_arguments(output, aliases):
    """Translate only exact attachment aliases; preserve unrelated or unknown arguments."""
    value=json.loads(json.dumps(output));rewrites=[]
    for call in value['tool_calls']:
        args=json.loads(call['arguments_json'])
        for field in ['image_path','image_paths']:
            if field not in args:continue
            old=args[field];items=old if isinstance(old,list) else [old]
            translated=[aliases.get(x,x) if isinstance(x,str) else x for x in items]
            for before,after in zip(items,translated):
                if before!=after:rewrites.append({'field':field,'local_alias':before,'canonical_path':after})
            args[field]=translated if isinstance(old,list) else translated[0]
        call['arguments_json']=json.dumps(args)
    return value,rewrites

def execute(job, local_root):
    budget = json.loads((local_root/'budget.json').read_text())
    counter = local_root/'private/call_count.json'
    used = json.loads(counter.read_text()) if counter.exists() else 1  # Initial verified CLI smoke.
    if used >= budget['max_subscription_calls'] or time.time() >= budget['hard_deadline_epoch']:
        return {'status':'budget_or_deadline','seconds':0}
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
        canonical={part['index']:part['path'] for m in job['messages'] if isinstance(m.get('content'),list) for part in m['content'] if part.get('type')=='image' and 'path' in part}
        aliases={}
        for i, encoded in enumerate(job.get('images',[])):
            image=p/f'i{i}.png'; image.write_bytes(base64.b64decode(encoded)); cmd+=['--image',str(image)]
            if i in canonical:aliases[str(image)]=canonical[i]
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
            result['raw_model_output']=value
            result['output'],result['transport_path_rewrites']=normalize_image_arguments(value,aliases)
        return result
