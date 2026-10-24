"""Single-consumer private queue; root supplied by a neutral deployment wrapper."""
import json
import os
from pathlib import Path
import re
import sys
ROOT=Path(os.environ['R997_ROOT'])

def main():
    request=json.load(sys.stdin);action=request['action'];queue=ROOT/'private/queue'
    if action=='pull':
        terminal=any((ROOT/x).exists() for x in ['FINISHED.json','FAILED.json','STOPPED.json'])
        if terminal:return {'state':'finished'}
        pending=[p for p in sorted(queue.glob('*.request.json')) if not p.with_name(p.name.replace('.request.','.response.')).exists()]
        if pending:
            p=pending[0];return {'state':'job','key':p.name.split('.')[0],'job':json.loads(p.read_text())}
        return {'state':'waiting'}
    if action=='respond':
        value=request['payload'];key=value['key'];assert re.fullmatch(r'[0-9a-f]{32}',key)
        p=queue/(key+'.response.json');tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value['response']));tmp.replace(p)
        source=queue/(key+'.request.json');job=json.loads(source.read_text());job['image_count']=len(job.get('images',[]));job.pop('images',None);tmp=source.with_suffix('.tmp');tmp.write_text(json.dumps(job));tmp.replace(source)
        if value['response']['status']=='budget_or_deadline':(ROOT/'BUDGET_EXHAUSTED.json').write_text(json.dumps({'key':key,'status':'budget_or_deadline'}))
        return {'saved':True}
    raise ValueError('Unknown action')

if __name__=='__main__':print(json.dumps(main()))
