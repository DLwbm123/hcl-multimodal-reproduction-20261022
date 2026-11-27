import sys
from collections import Counter, defaultdict
import binding
from binding import prior, e


def prepare():
    cases = []
    for index,visit in enumerate(e.read('external')):
        rows = visit['rows']
        days = sorted({day for day,_,_ in rows})
        for day in days:
            def avg(start,end,code=None):
                vals = [v for t,c,v in rows if start <= t <= end and (code is None or c==code)]
                return sum(vals)/len(vals) if vals else None
            def diff(a,b):
                return None if a is None or b is None else a-b
            am = avg(day-6,day,58)
            pm = avg(day-6,day,62)
            for k,y in enumerate([am,diff(avg(day-6,day),avg(day-13,day-7)),diff(pm,am)]):
                case = dict(id=str(len(cases)),person=visit['person'],visit=index,day=day,k=k,truth=y)
                e.canonical(case,rows)
                cases.append(case)
    assert len({c['person'] for c in cases})==25
    e.save('target',cases)
    print('verified',len(cases),flush=True)


def artifacts(stage):
    _,_,bank = binding.split()
    return e.read('target'),e.read('external'),bank,binding.learn(bank)


def issue(session,library):
    note = binding.issue(session,library)
    return note if note and note.startswith('Reversed operands.') else None


def report():
    cases = {c['id']:c for c in e.read('target')}
    rows = [r for i in range(4) for r in e.read('eval'+str(i))]
    assert len(rows)==len(cases) and {r['id'] for r in rows}==cases.keys()
    groups = defaultdict(list)
    for r in rows:
        c=cases[r['id']]
        for group in ('all','triggered' if r['reason'] else 'untriggered',
                      ('morning','weeks','windows')[c['k']],'numeric' if c['truth'] is not None else 'missing'):
            groups[group].append(r)
    stats = {}
    for group,subset in groups.items():
        s = dict(n=len(subset),patients=len({cases[r['id']]['person'] for r in subset}),
                 trigger=sum(bool(r['reason']) for r in subset))
        for arm in ('base','generic','attributed'):
            s[arm] = {k:sum(r[arm][k] for r in subset) for k in ('hit','calls','steps','raw')}
        for src,dst in [('base','generic'),('base','attributed'),('generic','attributed')]:
            patients = Counter()
            for r in subset:
                patients[cases[r['id']]['person']] += int(r[dst]['hit']) - int(r[src]['hit'])
            s[dst+'_vs_'+src] = dict(prior.paired(subset,src,dst),patients=dict(Counter(
                'gain' if d>0 else 'loss' if d<0 else 'tie' for d in patients.values())))
        stats[group]=s
    e.save('result',stats)


if __name__=='__main__':
    if sys.argv[1]=='prepare':
        prepare()
        assert len(artifacts('eval')[3])==3
    elif sys.argv[1]=='eval':
        prior.artifacts,prior.issue=artifacts,issue
        prior.inference('eval')
    else:
        report()
