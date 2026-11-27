"""Paired continuation of frozen first FHIR tool calls on the development partition."""
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import compare as v

p, d = v.p, v.d
archive = Path('archive')


def load(name):
    return json.loads((archive / (name + '.json')).read_text())


def mismatch(case, example, action):
    match = re.search(r'(?m)^GET (\S+)', example['demo'])
    patients = set(re.findall(r'\bS\d+\b', case['instruction']))
    if not match or '<MRN>' not in match[1] or len(patients) != 1 or not action.startswith('GET '):
        return []
    desired = urlsplit(match[1].replace('<MRN>', next(iter(patients))))
    actual = urlsplit(action[4:].strip())
    fields = []
    if actual.path != desired.path:
        fields.append(('resource',desired.path))
    params = parse_qs(actual.query)
    for key, required in parse_qs(desired.query).items():
        if key in ('patient','identifier','code') and params.get(key) != required:
            fields.append((key,required[0]))
    return fields


def read_ok(case, history):
    if d.kind(case) in d.WRITES:
        return None
    expected = urlsplit(d.query(case))
    needed = parse_qs(expected.query)
    for (role,action),(nrole,returned) in zip(history,history[1:]):
        if role != 'agent' or not action.startswith('GET ') or nrole != 'user' or not returned.startswith('{"returned":'):
            continue
        actual = urlsplit(action[4:].strip())
        if (actual.netloc,actual.path)==(expected.netloc,expected.path) and all(
                parse_qs(actual.query).get(k)==val for k,val in needed.items()):
            return True
    return False


def metrics(case, history, qualified, official):
    read = read_ok(case,history)
    return dict(hit=bool(qualified),official=bool(official),strict=bool(qualified and read is not False),
                read=read,calls=sum(role=='agent' and action.startswith(('GET ','POST ')) for role,action in history))


def continuation(case, prefix, prompt, first_read, feedback):
    history = [list(item) for item in prefix]
    reads = [first_read] if first_read is not None else []
    chat = [dict(role='user',content=prompt),dict(role='assistant',content=prefix[0][1]),
            dict(role='user',content=prefix[1][1]+'\n'+feedback)]
    answer = None
    for _ in range(7):
        action = p.generate([chat],limit=384)[0].strip().replace('```tool_code','').replace('```','').strip()
        history.append(['agent',action])
        chat.append(dict(role='assistant',content=action))
        if action.startswith('FINISH(') and action.endswith(')'):
            answer = action[7:-1]
            break
        if action.startswith('GET '):
            response,read = d.model_fetch(action[4:].strip())
            if read is not None:
                reads.append(read)
        elif action.startswith('POST '):
            try:
                json.loads(action.partition('\n')[2])
                response = 'POST request accepted and executed successfully.'
            except ValueError:
                response = 'Invalid POST request'
        else:
            break
        history.append(['user',response])
        chat.append(dict(role='user',content=response))
    official, qualified = p.grade(case,history,answer,reads)
    return metrics(case,history,qualified,official)


def check_receipt(saved,live):
    if saved.startswith('{') and live.startswith('{'):
        assert json.loads(saved)==json.loads(live), 'Historical and live receipt differ'
    else:
        assert saved==live, 'Historical and live response differ'


def run(part):
    v.setup_model()
    bank = load('library')
    partition = load('partition')
    train = {v.ROWS[i]['eval_MRN'] for i in partition['train']}
    assert train.isdisjoint(v.ROWS[i]['eval_MRN'] for i in partition['valid'])
    ids = partition['valid'][part::4]
    path = Path(f'part{part}.json')
    saved = json.loads(path.read_text()) if path.exists() else []
    for backbone in ('card','plain'):
        base = {r['id']:r for i in range(5) for r in load(f'valid_{backbone}_{i}')}
        assert set(base)==set(partition['valid'])
        for case_id in ids:
            if any(r['id']==case_id and r['backbone']==backbone for r in saved):
                continue
            b = base[case_id]
            c = v.ROWS[case_id]
            example = v.match(c,bank)
            h = b['history']
            fields = mismatch(c,example,h[0][1])
            start = metrics(c,h,b['qualified'],b['official'])
            row = dict(id=case_id,backbone=backbone,fields=[k for k,_ in fields],base=start)
            if fields:
                assert len(h)>=2 and h[1][0]=='user'
                live,read = d.model_fetch(h[0][1][4:].strip())
                check_receipt(h[1][1],live)
                prompt = p.template.format(api_base=p.BASE.rstrip('/'),functions=p.docs,context=c['context'],question=c['instruction'])
                if backbone=='card':
                    assert len(p.tok.encode(example['card'],add_special_tokens=False))<=1024
                    prompt += '\nPrior workflow memory (use only if relevant): '+example['card']
                generic = 'Recheck the first query against the current question and the tool documentation. Continue with one action.'
                specific = 'The first query has a binding mismatch. Expected '+', '.join(k+'='+val for k,val in fields)+'. '+generic
                row['generic']=continuation(c,h[:2],prompt,read,generic)
                row['specific']=continuation(c,h[:2],prompt,read,specific)
            else:
                row['generic']=row['specific']=start
            saved.append(row)
            p.save(str(path),saved)
            print(part,backbone,len(saved),flush=True)


def report():
    rows=[r for i in range(4) for r in json.loads(Path(f'part{i}.json').read_text())]
    assert len(rows)==len({(r['id'],r['backbone']) for r in rows})==154
    groups=defaultdict(list)
    for r in rows:
        for group in ('all','triggered' if r['fields'] else 'untriggered',p.family(v.ROWS[r['id']])):
            groups[r['backbone']+'_'+group].append(r)
    stats={}
    for name,sub in groups.items():
        s=dict(n=len(sub),people=len({v.ROWS[r['id']]['eval_MRN'] for r in sub}),
               triggered=sum(bool(r['fields']) for r in sub),field_types=dict(Counter(f for r in sub for f in r['fields'])),
               query_tasks=sum(r['base']['read'] is not None for r in sub))
        for arm in ('base','generic','specific'):
            s[arm]={k:sum(r[arm][k] or 0 for r in sub) for k in ('hit','strict','official','read','calls')}
        for a,b in [('base','specific'),('base','generic'),('generic','specific')]:
            data={}
            for k in ('hit','strict','read'):
                pairs=[(r[a][k],r[b][k]) for r in sub if r[a][k] is not None]
                people=Counter()
                for r in sub:
                    if r[a][k] is not None:
                        people[v.ROWS[r['id']]['eval_MRN']]+=int(r[b][k])-int(r[a][k])
                data[k]=dict(gain=sum(not x and y for x,y in pairs),loss=sum(x and not y for x,y in pairs),
                             retained=sum(x and y for x,y in pairs),
                             patients=dict(Counter('gain' if x>0 else 'loss' if x<0 else 'tie' for x in people.values())))
            s[b+'_vs_'+a]=data
        stats[name]=s
    p.save('result.json',stats)


if __name__=='__main__':
    if sys.argv[1]=='report':
        report()
    else:
        run(int(os.environ['PART']))
