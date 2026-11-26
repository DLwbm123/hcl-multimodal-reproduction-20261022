import json
from pathlib import Path
import b

def read(name):
    return json.loads(Path(name+'.json').read_text())

def main():
    b.main(validation_start=0, mrn_base=49100100)
    dev, resources = read('cases'), read('resources')
    counts = {}
    for case in dev:
        group = case['group']
        counts[group] = counts.get(group,0)+1
        case['stage'] = 'build' if counts[group]<=16 else 'gate'
    assert sorted(counts.values()) == [24,24,24]
    b.main(validation_start=72, mrn_base=49200100)
    test = read('cases')
    for case in test:
        case['stage'] = 'test'
    for i,case in enumerate(dev+test):
        case['id'] = f'task9_{i}'
    Path('cases.json').write_text(json.dumps(dev+test))
    Path('resources.json').write_text(json.dumps(resources+read('resources')))

if __name__ == '__main__':
    main()
