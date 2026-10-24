"""Small stdin RPC for a single sequential controller; no credentials accepted."""
import json
import os
from pathlib import Path
import re
import sys
ROOT=Path('/data/bmw/HCL/medrax_subscription_v4')

def main():
    request=json.load(sys.stdin); action=request['action']
    queue=ROOT/'private/queue'; queue.mkdir(exist_ok=True,parents=True)
    if action=='pull':
        pending=[p for p in sorted(queue.glob('*.request.json')) if not p.with_name(p.name.replace('.request.','.response.')).exists()]
        if pending:
            p=pending[0]; return {'state':'job','key':p.name.split('.')[0],'job':json.loads(p.read_text())}
        return {'state':'finished' if ((ROOT/'FINISHED.json').exists() or (ROOT/'FAILED.json').exists()) else 'waiting'}
    if action=='respond':
        p=request['payload']; key=p['key']; assert re.fullmatch(r'[0-9a-f]{32}',key)
        target=queue/(key+'.response.json'); temp=target.with_suffix('.tmp')
        temp.write_text(json.dumps(p['response'])); temp.replace(target)
        source=queue/(key+'.request.json')
        job=json.loads(source.read_text()); job['image_count']=len(job.get('images',[])); job.pop('images',None)
        temp=source.with_suffix('.tmp');temp.write_text(json.dumps(job));temp.replace(source)
        return {'saved':True}
    if action=='refresh_urls':
        p=ROOT/'private/download.json'; plan=json.loads(p.read_text())
        urls={(i['directory'],i['name']):i['url'] for i in request['payload']}
        for item in plan['files']:
            if (item['directory'],item['name']) in urls:item['url']=urls[(item['directory'],item['name'])]
        t=p.with_suffix('.tmp');t.write_text(json.dumps(plan));t.chmod(0o600);t.replace(p)
        return {'updated':len(urls)}
    raise ValueError('Unknown RPC action')
if __name__=='__main__': print(json.dumps(main()))
