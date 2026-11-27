"""Second shared-note edit on an existing, patient-disjoint two-submit task."""
import collections
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import prior as q

p, d = q.p, q.d
MENU = ("A get latest potassium; B calculate simulated replacement from the latest receipt; "
        "C submit calculated oral replacement; D submit next-morning potassium recheck; "
        "E get mean potassium; F finish with last statistic; G finish with empty list.")
PROMPT = "You control an offline simulator. Choose just one action letter. " + MENU
TASK = ("Check the latest potassium. If present and below 3.5, calculate the benchmark replacement, "
        "submit it, then submit the requested next-morning recheck. Otherwise submit nothing. End with [].")


def prepare():
    cases = q.read('source/cases')
    timeline = collections.defaultdict(list)
    for obj in q.read('source/resources'):
        if obj['resourceType'] == 'Observation':
            who = obj['subject']['reference'].split('/')[-1]
            timeline[who].append([obj['effectiveDateTime'], obj['valueQuantity']['value']])
    data = []
    for c in cases:
        rows = sorted(timeline[c['eval_MRN']])
        assert [x for _, x in rows] == c['source_values']
        data.append(dict(id=c['eval_MRN'], rows=rows, stage={'build':'train', 'gate':'gate', 'test':'test'}[c['stage']]))
    assert len(data) == len({c['id'] for c in data}) == 144
    p.save('kdata.json', data)


def branch(c):
    if not c['rows']:
        return 'missing'
    return 'low' if c['rows'][-1][1] < 3.5 else 'other'


def reference(task, c):
    return list('ABCDG' if branch(c) == 'low' else 'AG')


def submit(c, action, dose):
    obj = dict(status='active', intent='order', authoredOn=d.MONITOR.isoformat(), subject={'reference':'Patient/'+c['id']})
    if action == 'C':
        obj.update(resourceType='MedicationRequest', medicationCodeableConcept={'coding':[dict(system='http://hl7.org/fhir/sid/ndc',code='40032-917-01')]},
                   dosageInstruction=[dict(route='oral', doseAndRate=[dict(doseQuantity=dict(value=dose,unit='mEq'))])])
    else:
        obj.update(resourceType='ServiceRequest', code={'coding':[dict(system='http://loinc.org',code='2823-3')]},
                   priority='stat', occurrenceDateTime='2023-11-14T08:00:00+00:00')
    return obj


def step(c, s, action):
    s['commands'].append(action)
    latest, mean = q.facts(c)
    if action in 'AE':
        s['seen'].add(action)
        s['last'] = latest if action == 'A' else mean
        return 'Statistic: '+str(s['last'])
    if action == 'B':
        if 'A' not in s['seen'] or not c['rows']:
            return 'Cannot calculate without a measured latest receipt.'
        s['dose'] = round(max(0, (3.5-latest)*100), 3)
        return 'Calculated simulated replacement: '+str(s['dose'])
    if action in 'CD':
        if action == 'C' and not s.get('dose'):
            return 'No positive calculated replacement to submit.'
        s['posts'].append(submit(c, action, s.get('dose')))
        return 'Simulated request accepted: '+s['posts'][-1]['resourceType']
    if action == 'F' and s['last'] is None:
        return 'Obtain a statistic first.'
    s['done'] = True
    s['answer'] = [] if action == 'G' else [s['last']]
    return 'Finished.'


def score(task, c, s):
    posts = s['posts']
    received = [('Observation', {'patient':[c['id']], 'code':['K']})]
    entries = [{'resource':{'effectiveDateTime':stamp,'valueQuantity':{'value':value}}} for stamp,value in c['rows']]
    expected_url = p.BASE+'Observation?patient='+c['id']+'&code=K&_count=5000&_format=json'
    def cached(url):
        assert url == expected_url
        return {'data':json.dumps({'entry':entries}),'status_code':200}
    hist = []
    for obj in posts:
        hist.extend([('agent','POST '+p.BASE+obj['resourceType']+'\n'+json.dumps(obj)), ('user','POST request accepted and executed successfully.')])
    hist.append(('agent','FINISH([])'))
    with patch.object(p.ref, 'send_get_request', cached):
        valid = p.grade({'id':'task9_control','eval_MRN':c['id']}, hist, '[]', received)[1]
    seq = ''.join(s['commands'])
    first_submit = next((i for i,v in enumerate(seq) if v in 'CD'), len(seq))
    supported = 'A' in seq[:first_submit]
    write = branch(c) == 'low'
    return dict(id=c['id'], hit=bool(valid and supported and s['done'] and s['answer']==[]), write=write,
                correct_write=bool(write and valid and supported), extra=bool(posts) and not write, seq=seq,
                calls=1+sum(a in 'ABCDE' for a in seq))


def question(task, c):
    return TASK+'\nCurrent identifier: '+c['id']+'\nRecords (time, mmol/L): '+json.dumps(c['rows'], separators=(',',':'))


class Controller(q.Runner):
    def run(self, task, split, note):
        if task != 'K':
            return super().run(task, split, note)
        data = q.read('kdata')
        with patch.object(self,'data',data), patch.object(self,'train',[c for c in data if c['stage']=='train']), \
             patch.multiple(q, DOC=PROMPT, question=question, reference=reference, step=step, score=score):
            return super().run(task, split, note)


def proposals(old, rows):
    cases = {c['id']:c for c in q.read('kdata') if c['stage']=='train'}
    failures = [r for r in rows if not r['hit']]
    prompt = ("Write a short reusable English procedural note for another agent. You are a document editor. "
              "Explain in ordinary sentences which observation to read, when each branch applies and when to stop. "
              "The simulator's menu is quoted here: "+MENU+"\nAssigned workflow: "+TASK+
              "\nPrior shared note: "+old+"\nBuild failures: "+str(len(failures))+"/"+str(len(rows))+
              "; failed low cases: "+str(sum(r['write'] for r in failures))+"; unnecessary orders: "+str(sum(r['extra'] for r in rows)))
    counts = collections.Counter((r['seq'], ''.join(reference('K',cases[r['id']]))) for r in failures)
    choices = {'unchanged':old}
    for label, suffix in [('aggregate',''),('trajectory','\nObserved versus verified letter sequences: '+str(counts.most_common(10)))]:
        note = p.generate([[dict(role='system',content='You edit workflow manuals. Return the requested prose.'),
                            dict(role='user',content=prompt+suffix)]],limit=160)[0].strip()
        if len(note.split()) >= 4:
            choices[label] = note
    table = collections.defaultdict(collections.Counter)
    for c in cases.values():
        table[branch(c)][' '.join(reference('K',c))] += 1
    choices['induced'] = 'For the potassium workflow only, after the initial observation use these sequences: '+', '.join(
        label+' -> '+table[group].most_common(1)[0][0] for group,label in [('low','measured below 3.5'),('other','measured at least 3.5'),('missing','missing')])+'.'
    return choices


def aggregate(rows):
    return {k:sum(r[k] for r in rows) for k in ('hit','correct_write','write','extra','calls')}


def workflow():
    q.v.setup_model()
    r = Controller()
    old = q.read('parent/aw')['note']
    r.stage('build','K','train',[old])
    if r.j == 0:
        p.save('choices.json',proposals(old,q.merge('build')[0]))
    candidates = q.wait('choices')
    labels, notes = list(candidates), list(candidates.values())
    for task in ('A','K'):
        r.stage('gate'+task,task,'gate',notes)
    if r.j == 0:
        new = [len(q.successes(x)) for x in q.merge('gateK')]
        old_sets = [q.successes(x) for x in q.merge('gateA')]
        h0 = q.successes(q.merge('parent/ag')[0])
        gained = old_sets[0]-h0
        eligible = [i for i,x in enumerate(old_sets) if gained<=x and len(x)>=len(old_sets[0])]
        p.save('selected.json',dict(normal=labels[max(range(len(labels)),key=new.__getitem__)],
                                    protected=labels[max(eligible,key=new.__getitem__)],new=new,old=list(map(len,old_sets)),
                                    eligible=[labels[i] for i in eligible],gain_n=len(gained)))
    selected = q.wait('selected')
    for task in ('A','K'):
        r.stage('test'+task,task,'test',['']+notes)
    if r.j: return
    result = dict(selection=selected, candidates={k:{'words':len(val.split()),'same_as_old':val==old}
                                                  for k,val in candidates.items()})
    for task in ('A','K'):
        rows = dict(zip(['fixed']+labels,q.merge('test'+task)))
        n = 180 if task=='A' else 72
        assert all(len(rr)==len({x['id'] for x in rr})==n for rr in rows.values())
        acquired = q.successes(rows['unchanged'])-q.successes(rows['fixed'])
        result[task] = {k:dict(**aggregate(rr), prior=q.paired(rows['unchanged'],rr), fixed=q.paired(rows['fixed'],rr),
                               retained_new=len(acquired & q.successes(rr))) for k,rr in rows.items()}
    controls = []
    for c in q.read('kdata'):
        if c['stage']=='test':
            s=q.state(c)
            for a in reference('K',c): step(c,s,a)
            controls.append(score('K',c,s))
    assert aggregate(controls)['hit']==72
    result['rule_K']=aggregate(controls)
    p.save('result.json',result)


if __name__=='__main__':
    if sys.argv[1]=='prepare':
        prepare()
    else:
        try: workflow()
        except BaseException:
            Path('failed'+os.environ['PART']).touch()
            raise
