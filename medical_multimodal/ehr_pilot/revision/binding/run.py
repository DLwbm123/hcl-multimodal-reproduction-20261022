import collections
import json
import math
import random
import sys

import prior
from prior import e


def split():
    cases, assessment = prior.source_split()
    library_ids = {c['id'] for c in cases if c['person'] not in assessment}
    examples = {i: b for i, b in e.read('source/bank').items() if i in library_ids}
    return cases, assessment, examples


def prepare():
    previous, people, _ = split()
    visit_ids = sorted({c['visit'] for c in previous if c['person'] in people})
    visits = e.read('source/visits')
    cases = []
    for index in visit_ids:
        visit = visits[index]
        rows = visit['rows']
        days = sorted({day for day, _, _ in rows})
        for day in days:
            means = {}
            for name, code, start, end in [('am',58,day-6,day), ('pm',62,day-6,day),
                                           ('recent',None,day-6,day), ('prev',None,day-13,day-7)]:
                values = [v for t,c,v in rows if start <= t <= end and (code is None or c == code)]
                means[name] = sum(values)/len(values) if values else None
            def diff(a,b):
                return None if means[a] is None or means[b] is None else means[a]-means[b]
            for k,y in enumerate([means['am'],diff('recent','prev'),diff('pm','am')]):
                case = dict(id=str(len(cases)), person=visit['person'], visit=index, day=day, k=k, truth=y)
                e.canonical(case,rows)
                cases.append(case)
    assert len(cases)==1194 and len(people)==28
    e.save('target', cases)
    print(len(cases),flush=True)


def receipts(history):
    handles, counts, means = {}, {}, {}
    operands = None
    for (role, action), (next_role, output) in zip(history,history[1:]):
        if role != 'assistant' or next_role != 'user' or not output.startswith('{'):
            continue
        rec = json.loads(output)
        cmd = action.split()
        if cmd[0]=='READ' and 'handle' in rec:
            key = tuple(cmd[1:])
            handles[rec['handle']] = key
            counts[key] = rec['count']
        elif cmd[0]=='STAT' and cmd[-1]=='mean' and rec.get('value') is not None:
            means[handles[cmd[1]]] = rec['value']
        elif cmd[0]=='DIFF' and rec.get('value') is not None:
            operands = (float(cmd[1]),float(cmd[2]))
    return handles, counts, means, operands


def close(a,b):
    return math.isclose(a,b,rel_tol=0,abs_tol=0.05)


def learn(examples):
    observed = collections.defaultdict(set)
    for ex in examples.values():
        messages = [(m['role'], m['content']) for m in ex['roles']]
        if messages[-1][1]=='FINISH NA':
            continue
        _,_,means,args = receipts(messages)
        if args is None:
            assert len(means)==1
            order = tuple(means)
        else:
            matches = [tuple(k for k,v in means.items() if close(a,v)) for a in args]
            if any(len(keys)!=1 for keys in matches):
                continue
            order = tuple(keys[0] for keys in matches)
            if len(set(order))!=2:
                continue
        observed[ex['k']].add(order)
    assert set(observed)=={0,1,2} and all(len(v)==1 for v in observed.values())
    return {k:next(iter(v)) for k,v in observed.items()}


def artifacts(stage):
    _,_,examples = split()
    return e.read('target'),e.read('source/visits'),examples,learn(examples)


def issue(session,library):
    keys = library[session.c['k']]
    handles,counts,means,operands = receipts(session.h)
    if session.value is None:
        if any(counts.get(key)==0 for key in keys):
            return None
        by_key = {key:h for h,key in handles.items()}
        for key in keys:
            if key not in counts:
                return 'Missing evidence. Use READ '+' '.join(key)+' for the current patient.'
            if key not in means:
                return 'Missing evidence. Use STAT '+by_key[key]+' mean on the nonempty current group.'
    elif isinstance(session.value,float) and len(keys)==2 and operands and all(k in means for k in keys):
        a,b = (means[k] for k in keys)
        first,second = operands
        reverse = close(first,b) and close(second,a)
        forward = close(first,a) and close(second,b)
        if reverse and not forward:
            return ('Reversed operands. Run DIFF using the mean for '+' '.join(keys[0])+
                    ' first and the mean for '+' '.join(keys[1])+' second, from current receipts.')
    return None


def reason(row):
    msg = row['reason'] or ''
    return 'reversed' if msg.startswith('Reversed') else 'missing' if msg else 'untriggered'


def report():
    cases = {c['id']:c for c in e.read('target')}
    rows = [r for i in range(4) for r in e.read('eval'+str(i))]
    assert len(rows)==len(cases)==len({r['id'] for r in rows}) and {r['id'] for r in rows}==cases.keys()
    people = sorted({c['person'] for c in cases.values()})
    random.Random(26112712).shuffle(people)
    fold = {p:i//14 for i,p in enumerate(people)}
    gates = {}
    for f in (0,1):
        gates[f] = {}
        for label in ('reversed','missing'):
            other = [r for r in rows if fold[cases[r['id']]['person']]!=f and reason(r)==label]
            pair = prior.paired(other,'base','attributed')
            gates[f][label] = dict(pair,enabled=pair['gain']>0 and pair['loss']==0)
    groups = collections.defaultdict(list)
    for row in rows:
        c = cases[row['id']]
        label = reason(row)
        active = gates[fold[c['person']]].get(label,{}).get('enabled',False)
        row['selected'] = row['attributed'] if active else row['base']
        for g in ['all',label,('morning','weeks','windows')[c['k']], 'missing_answer' if c['truth'] is None else 'numeric_answer']:
            groups[g].append(row)
    stats = {}
    for g, rr in groups.items():
        subject = lambda r: cases[r['id']]['person']
        arms = ('base','generic','attributed','selected')
        s = dict(n=len(rr),people=len({subject(r) for r in rr}),triggered=sum(bool(r['reason']) for r in rr))
        s['arms'] = {a:{k:sum(r[a][k] for r in rr) for k in ('hit','raw','steps','calls')} for a in arms}
        for a,b in [('base','generic'),('base','attributed'),('generic','attributed'),('base','selected')]:
            delta = collections.Counter()
            for r in rr:
                delta[subject(r)] += int(r[b]['hit'])-int(r[a]['hit'])
            s[b+'_vs_'+a] = dict(prior.paired(rr,a,b),patient_direction=dict(collections.Counter(
                'gain' if v>0 else 'loss' if v<0 else 'tie' for v in delta.values())))
        stats[g] = s
    e.save('result',dict(gates=gates,stats=stats))


if __name__=='__main__':
    if sys.argv[1]=='prepare':
        prepare()
        assert len(artifacts('eval')[3])==3
    elif sys.argv[1]=='eval':
        prior.artifacts,prior.issue = artifacts,issue
        prior.inference('eval')
    else:
        report()
