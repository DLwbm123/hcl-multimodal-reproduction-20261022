"""Post hoc receipt and next-action checks. Output contains counts only."""
import json
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit

def read(file):
    return json.loads(Path(file).read_text())

cases = {case['id']: case for case in read('cases.json')}
result = {}
for arm in ('original', 'lookup', 'edit'):
    counts = defaultdict(Counter)
    for shard in range(3):
        for row in read(f'{arm}_{shard}.json'):
            case = cases[row['id']]
            group = counts[case['group']]
            history = row['history']
            receipt = json.loads(history[1][1])
            series = receipt['series']
            values = [value for item in series for _, value in item['measurements']]
            scoped = all(item['subject'] == 'Patient/' + case['eval_MRN'] for item in series)
            group['receipt_matches_source'] += scoped and receipt['returned'] == len(values) and values == case['source_values'][::-1]
            action = history[2][1]
            if action.startswith('POST '):
                label = 'POST ' + urlsplit(action.splitlines()[0][5:]).path.rsplit('/', 1)[-1]
            elif action.startswith('FINISH(') and action.endswith(')'):
                label = 'FINISH'
            else:
                label = ('IF' if action.startswith('IF ') else 'LOG' if action.startswith('LOG ') else 'other')
            group[label] += 1
            group['official_success'] += row['official']
    result[arm] = counts
print(json.dumps(result, indent=2))
