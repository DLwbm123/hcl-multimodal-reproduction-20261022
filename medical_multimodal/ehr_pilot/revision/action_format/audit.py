"""Report only counts for the first action after retrieval."""
import json
from collections import Counter
from pathlib import Path

def read(name):
    return json.loads(Path(name+'.json').read_text())

cases={c['id']:c for c in read('cases')}
stats={}
for arm in ('plain','constrained'):
    stats[arm]={}
    for part in range(3):
        for row in read(f'{arm}_{part}'):
            group=stats[arm].setdefault(cases[row['id']]['group'],Counter())
            first=row['history']
            values=[value for series in json.loads(first[1][1])['series'] for _,value in series['measurements']]
            group['source_agreement']+=values==cases[row['id']]['source_values'][::-1]
            act=first[2][1]
            if act.startswith('POST '):
                endpoint=act.partition('\n')[0][5:]
                if endpoint.startswith('http://127.0.0.1:18478/fhir/'):
                    group['local_endpoint_prefix']+=1
                elif endpoint.strip().startswith('http://127.0.0.1:18478/fhir/'):
                    group['leading_whitespace']+=1
                try:
                    json.loads(act.partition('\n')[2])
                    key='parseable_post'
                except ValueError:
                    key='malformed_post'
            elif act.startswith('FINISH(') and act.endswith(')'):
                try:
                    answer=json.loads(act[7:-1])
                    key='empty_answer' if answer==[] else 'nonempty_answer'
                except ValueError:
                    key='malformed_finish'
            else:
                key='other'
            group[key]+=1
print(json.dumps(stats,indent=2))
