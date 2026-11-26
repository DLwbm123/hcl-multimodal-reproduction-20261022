import json
import os
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import torch
import compare as v
from transformers import LogitsProcessorList

def read(name):
    return json.loads(Path(name + '.json').read_text())

def eligible(chat):
    if len(chat) != 3 or not chat[-1]['content'].startswith('{"returned":'):
        return False
    action = chat[1]['content']
    if not action.startswith('GET '):
        return False
    url = urlsplit(action[4:].strip())
    return url.path == '/fhir/Observation' and parse_qs(url.query).get('code') == ['K']

def install_constraint():
    plain = v.p.generate
    generate = v.p.model.generate
    paths = [v.p.tok.encode(prefix, add_special_tokens=False) for prefix in ('GET ', 'POST ', 'FINISH(')]
    max_len = max(map(len, paths))

    def constrained(chats, *args, **kwargs):
        active = [i for i, chat in enumerate(chats) if eligible(chat)]
        if not active:
            return plain(chats, *args, **kwargs)

        def with_mask(*model_args, **model_kwargs):
            width = model_kwargs['input_ids'].shape[1]
            def mask(tokens, scores):
                step = tokens.shape[1] - width
                if step >= max_len:
                    return scores
                for i in active:
                    prefix = tokens[i, width:].tolist()
                    if prefix in paths:
                        continue
                    allowed = [path[step] for path in paths if len(path) > step and path[:step] == prefix]
                    # A shorter completed action prefix can have free text already.
                    if not allowed:
                        assert any(prefix[:len(path)] == path for path in paths)
                        continue
                    kept = scores[i, allowed].clone()
                    scores[i].fill_(-torch.inf)
                    scores[i, allowed] = kept
                return scores
            return generate(*model_args, **model_kwargs, logits_processor=LogitsProcessorList([mask]))
        v.p.model.generate = with_mask
        try:
            return plain(chats, *args, **kwargs)
        finally:
            v.p.model.generate = generate
    v.p.generate = constrained

def executable(action):
    if action.startswith('FINISH(') and action.endswith(')'):
        return True
    if action.startswith('GET '):
        return action.count('\n') == 0 and action[4:].startswith(v.p.BASE)
    if action.startswith('POST '):
        try:
            json.loads(action.partition('\n')[2])
            return action.splitlines()[0][5:].startswith(v.p.BASE)
        except ValueError:
            pass
    return False

def main():
    cases = read('cases')
    if sys.argv[1] == 'infer':
        assert read('controls')['checked'] == 72
        v.setup_model()
        idx = int(os.environ['PART'])
        for arm in ('plain', 'constrained'):
            if arm == 'constrained':
                install_constraint()
            v.p.run(cases[idx::3], f'{arm}_{idx}', lambda c: read('edit_9')['candidate'], batch_size=2, hint_budget=1024)
        return
    data = {}
    for arm in ('plain','constrained'):
        rows = [row for ix in range(3) for row in read(f'{arm}_{ix}')]
        data[arm] = {r['id']:r for r in rows}
        assert len(rows) == len(data[arm]) == 72
    counts = {}
    for c in cases:
        a,b = [data[k][c['id']] for k in ('plain','constrained')]
        assert a['history'][:2] == b['history'][:2]
        for group in ('all', c['group']):
            bucket = counts.setdefault(group, Counter())
            bucket.update(n=1,gain=int(b['qualified'] and not a['qualified']),loss=int(a['qualified'] and not b['qualified']),retained=int(a['qualified'] and b['qualified']))
            for arm,row in [('plain',a),('constrained',b)]:
                action = row['history'][2][1] if len(row['history'])>2 else ''
                bucket[arm+'_qualified'] += row['qualified']
                bucket[arm+'_official'] += row['official']
                bucket[arm+'_syntax'] += executable(action)
                bucket[arm+'_second_'+next((s for s in ('GET','POST','FINISH') if action.startswith(s)),'other')] += 1
    v.p.save('result.json', counts)

if __name__ == '__main__':
    main()
