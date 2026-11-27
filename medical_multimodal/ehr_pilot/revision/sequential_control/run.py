"""Three disjoint patient partitions. Frozen decoder and one shared, editable workflow note."""
import collections
import datetime as dt
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import compare as v

p,d=v.p,v.d
KINDS='ABCDEFG'
MENU=('A GET_LATEST_24H; B GET_MEAN_24H; C SUBMIT_1; D SUBMIT_2; E SUBMIT_4; '
      'F FINISH_LAST (return the last statistic as a one-element list); G FINISH_EMPTY (return []).')
DOC=('You are controlling a benchmark simulator. Review the current record. Use the tools for any required statistic. '
     'Every SUBMIT creates a simulated medication request for the current identifier, at the selected dose tier. '
     'Wait for the simulated receipt before ending. Each action consumes one step. Output just the letter for the next action. '+MENU)
ORDER=('Use GET_LATEST_24H to check the most recent magnesium. If it is missing or above 1.9, do not submit. '
       'Otherwise submit exactly one simulated request: dose 4 for a value below 1, dose 2 for a value below 1.5, otherwise dose 1.')
QUEST={'A':ORDER+' End with an empty list.',
       'B':'Use GET_MEAN_24H to report the mean magnesium over the past 24 hours, or the missing sentinel when the window has no observations. Do not submit.',
       'C':ORDER+' Also report the mean magnesium over the past 24 hours or the missing sentinel. End with the mean result.'}


def read(key):
    return json.loads(Path(key+'.json').read_text())


def prepare():
    patients=[c for c in read('cases_source/cases') if c['stage']=='build']
    people={c['eval_MRN'] for c in patients}
    assert len(patients)==len(people)==360
    resources=collections.defaultdict(list)
    for obj in read('cases_source/resources'):
        mrn=obj.get('subject',{}).get('reference','').removeprefix('Patient/')
        if mrn in people:
            t=dt.datetime.fromisoformat(obj['effectiveDateTime'])
            assert d.MONITOR-dt.timedelta(days=1)<=t<=d.MONITOR
            resources[mrn].append([obj['effectiveDateTime'],obj['valueQuantity']['value']])
    rng=random.Random(26112717)
    groups=collections.defaultdict(list)
    for c in patients:groups[c['group']].append(c)
    samples=[]
    for key in sorted(groups):
        group=sorted(groups[key],key=lambda c:c['eval_MRN'])
        assert len(group)==120
        rng.shuffle(group)
        for i,c in enumerate(group):
            points=sorted(resources[c['eval_MRN']])
            assert [x for _,x in points]==c['source_values']
            case=dict(id=c['eval_MRN'],stage='train' if i<40 else 'gate' if i<60 else 'test',rows=points)
            samples.append(case)
    p.save('dataset.json',samples)


def facts(case):
    values=[x for _,x in case['rows']]
    return (values[-1],sum(values)/len(values)) if values else (-1,-1)


def expected(t,case):
    value,_=facts(case)
    dose=None if t=='B' or not case['rows'] or value>1.9 else 4 if value<1 else 2 if value<1.5 else 1
    return dose


def orders(case,dose):
    return dict(resourceType='MedicationRequest',status='active',intent='order',subject={'reference':'Patient/'+case['id']},
                medicationCodeableConcept={'coding':[{'system':'http://hl7.org/fhir/sid/ndc','code':'0338-1715-40'}]},
                authoredOn=d.MONITOR.isoformat(),dosageInstruction=[{'route':'IV','doseAndRate':[{'doseQuantity':{'value':dose,'unit':'g'},
                'rateQuantity':{'value':dose,'unit':'h'}}]}])


def reference(t,c):
    seq=[] if t=='B' else ['A']
    dose=expected(t,c)
    if dose is not None:seq.append({1:'C',2:'D',4:'E'}[dose])
    return seq+(['B','F'] if t!='A' else ['G'])


def state(c):
    return dict(commands=[],posts=[],answer=None,last=None,stat=None,seen=set(),done=False)


def step(c,s,a):
    s['commands'].append(a)
    if a in 'AB':
        x,mean=facts(c)
        s['last']=x if a=='A' else mean
        s['stat']=a;s['seen'].add(a)
        return 'Statistic: '+json.dumps(s['last'])
    if a in 'CDE':
        payload=orders(c,{'C':1,'D':2,'E':4}[a])
        s['posts'].append(payload)
        return 'Simulated MedicationRequest accepted.'
    if a=='F' and s['last'] is None:return 'No statistic receipt. Obtain a statistic first.'
    s['answer']=[] if a=='G' else [s['last']]
    s['done']=True
    return 'Finished.'


def score(t,c,s):
    dose=expected(t,c)
    posts=s['posts']
    target_value=facts(c)[1]
    terminal=(s['done'] and (s['answer']==[] if t=='A' else len(s['answer'])==1 and abs(s['answer'][0]-target_value)<0.1 and s['stat']=='B'))
    values=[x['dosageInstruction'][0]['doseAndRate'][0]['doseQuantity']['value'] for x in posts]
    # Evaluate the simulator's actual submitted resources and dosage tier, including patient binding.
    post_ok=values==([] if dose is None else [dose]) and all(obj==orders(c,val) for obj,val in zip(posts,values))
    first_post=next((i for i,cmd in enumerate(s['commands']) if cmd in 'CDE'),len(s['commands']))
    supported=t=='B' or 'A' in s['commands'][:first_post]
    if t in ('A','C'):
        entries=[{'resource':{'effectiveDateTime':stamp,'valueQuantity':{'value':x}}} for stamp,x in c['rows']]
        url=p.BASE+'Observation?patient='+c['id']+'&code=MG&_count=5000&_format=json'
        def cached(query):
            assert query==url
            return {'data':json.dumps({'entry':entries}),'status_code':200}
        p.ref.send_get_request=cached
        hist=[]
        for obj in posts:
            hist.extend([('agent','POST '+p.BASE+'MedicationRequest'+'\n'+json.dumps(obj)),('user','POST request accepted and executed successfully.')])
        hist.append(('agent','FINISH([])'))
        official=p.grade({'id':'task5_control','eval_MRN':c['id']},hist,'[]',[('Observation',{'patient':[c['id']],'code':['MG']})])[0]
        assert official==post_ok, 'official write check disagrees with simulator'
    return dict(id=c['id'],hit=bool(terminal and post_ok and supported),write=dose is not None,
                correct_write=bool(dose is not None and post_ok and supported),extra=bool(posts) and dose is None,
                seq=''.join(s['commands']),calls=1+sum(a in 'ABCDE' for a in s['commands']))


def question(t,c):
    return QUEST[t]+'\nCurrent identifier: '+c['id']+'\nQueried observations (time, mg/dL): '+json.dumps(c['rows'],separators=(',',':'))


def example(t,c):
    q=[dict(role='user',content=question(t,c))]
    s=state(c)
    for a in reference(t,c):
        response=step(c,s,a);q.append(dict(role='assistant',content=a))
        q.append(dict(role='user',content=response))
    assert score(t,c,s)['hit']
    return q


class Runner:
    def __init__(self):
        self.data=read('dataset')
        self.train=[c for c in self.data if c['stage']=='train']
        self.j=int(os.environ['PART'])
        self.codes=[p.tok.encode(a,add_special_tokens=False) for a in KINDS]
        assert all(len(c)==1 for c in self.codes)
        self.cache={}
        self.examples={}
        self.base_chat={}

    def prefix(self,t,c,note):
        key=(t,c['id'])
        if key not in self.base_chat:
            ex=min((x for x in self.train if x['id']!=c['id']),key=lambda x:(abs(len(x['rows'])-len(c['rows'])),x['id']))
            chat=[]
            for task in ('AB' if t=='C' else t):
                ek=(task,ex['id'])
                if ek not in self.examples:self.examples[ek]=example(task,ex)
                chat.extend(self.examples[ek])
            self.base_chat[key]=chat
        doc=DOC+('\nShared workflow note: '+note if note else '')
        return [dict(role='system',content=doc)]+self.base_chat[key]+[dict(role='user',content=question(t,c))]

    @torch.inference_mode()
    def choose(self,chats):
        text=[p.tok.apply_chat_template(chat,tokenize=False,add_generation_prompt=True) for chat in chats]
        inputs=p.tok(text,padding=True,add_special_tokens=False,return_tensors='pt').to(p.model.device)
        logits=p.model(**inputs,use_cache=False).logits[:,-1,[t[0] for t in self.codes]]
        return [KINDS[i] for i in logits.argmax(-1).tolist()]

    def run(self,t,split,note):
        key=(t,split,note)
        if key in self.cache:return self.cache[key]
        cases=[c for c in self.data if c['stage']==split][self.j::4]
        out=[]
        for start in range(0,len(cases),8):
            batch=cases[start:start+8]
            states=[state(c) for c in batch]
            chats=[self.prefix(t,c,note) for c in batch]
            for _ in range(7):
                live=[i for i,s in enumerate(states) if not s['done']]
                if not live:break
                for i,cmd in zip(live,self.choose([chats[i] for i in live])):
                    response=step(batch[i],states[i],cmd)
                    chats[i].extend([dict(role='assistant',content=cmd),dict(role='user',content=response)])
            out.extend(score(t,c,s) for c,s in zip(batch,states))
        self.cache[key]=out
        return out

    def stage(self,key,t,split,notes):
        values=[self.run(t,split,note) for note in notes]
        p.save(key+str(self.j)+'.json',values)
        print(key,self.j,len(values[0]),flush=True)

    def edits(self,t,old,rows):
        by={c['id']:c for c in self.train}
        patterns=collections.Counter()
        for r in rows:
            if not r['hit']:
                patterns[(r['seq'],''.join(reference(t,by[r['id']])))] += 1
        if not patterns:return [old]
        base=('Edit the one shared workflow note to help the frozen action controller complete the task. '
              'Keep useful prior behavior. Use concise actionable rules. Output only the new note (under 140 tokens).\n'+
              DOC+'\nCurrent task: '+QUEST[t]+'\nOld note: '+(old or '(none)')+
              '\nActual complete successes: '+str(sum(r['hit'] for r in rows))+'/'+str(len(rows))+
              '. Failed cases requiring submission: '+str(sum(r['write'] and not r['hit'] for r in rows))+
              '. Unnecessary submissions: '+str(sum(r['extra'] for r in rows)))
        detail='\nObserved -> verified action sequences and counts: '+str(patterns.most_common(10))
        candidates=[old]
        for prompt in (base,base+detail):
            s=p.generate([[dict(role='user',content=prompt)]],limit=160)[0].strip()
            if s and s not in candidates:candidates.append(s)
        return candidates


def wait(key):
    while not Path(key+'.json').exists():
        if list(Path('.').glob('failed*')):raise RuntimeError('another shard failed')
        time.sleep(0.6)
    return read(key)


def merge(key):
    data=[wait(key+str(i)) for i in range(4)]
    return [sum([s[k] for s in data],[]) for k in range(len(data[0]))]


def paired(x,y):
    a={r['id']:r['hit'] for r in x};b={r['id']:r['hit'] for r in y}
    assert a.keys()==b.keys()
    return dict(gain=sum(not a[i] and b[i] for i in a),loss=sum(a[i] and not b[i] for i in a),retained=sum(a[i] and b[i] for i in a))


def successes(rows):
    return {r['id'] for r in rows if r['hit']}


def workflow():
    v.setup_model();r=Runner()
    r.stage('at','A','train',[''])
    if r.j==0:p.save('ae.json',r.edits('A','',merge('at')[0]))
    choices=wait('ae');r.stage('ag','A','gate',choices)
    if r.j==0:
        rows=merge('ag');best=max(range(len(choices)),key=lambda k:len(successes(rows[k])))
        p.save('aw.json',dict(index=best,note=choices[best],scores=list(map(lambda v:len(successes(v)),rows))))
    aw=wait('aw');a=aw['note']
    r.stage('bt','B','train',[a])
    if r.j==0:p.save('be.json',r.edits('B',a,merge('bt')[0]))
    choices=wait('be');r.stage('bg','B','gate',choices)
    r.stage('old','A','gate',choices)
    if r.j==0:
        new,old=merge('bg'),merge('old')
        new_scores=[len(successes(s)) for s in new];old_sets=[successes(s) for s in old]
        old_gain=old_sets[0]-successes(merge('ag')[0])
        eligible=[i for i,s in enumerate(old_sets) if old_gain<=s and len(s)>=len(old_sets[0])]
        p.save('bw.json',dict(normal=max(range(len(choices)),key=lambda i:new_scores[i]),
                              protected=max(eligible,key=lambda i:new_scores[i]),new_scores=new_scores,
                              old_scores=[len(s) for s in old_sets],gain_n=len(old_gain),eligible=eligible))
    bw=wait('bw')
    notes=['',a,choices[bw['normal']],choices[bw['protected']]]
    for task in 'ABC':r.stage('test'+task,task,'test',notes)
    if r.j:return
    result={}
    for task in 'ABC':
        arms=merge('test'+task)
        test=[c for c in r.data if c['stage']=='test'];solver=[]
        for c in test:
            s=state(c)
            for cmd in reference(task,c):step(c,s,cmd)
            solver.append(score(task,c,s))
        all_arms=arms+[solver]
        assert len(test)==180 and all(len(x)==180 and len({r['id'] for r in x})==180 for x in all_arms)
        stats=[]
        for arm in all_arms:
            stats.append({k:sum(r[k] for r in arm) for k in ('hit','write','correct_write','extra','calls')})
        acquired=successes(arms[1])-successes(arms[0])
        result[task]=dict(n=len(test),arms=dict(zip(['fixed','after_A','after_B','protected','rule'],stats)),
                          a_vs_fixed=paired(arms[0],arms[1]),b_vs_a=paired(arms[1],arms[2]),
                          protected_vs_a=paired(arms[1],arms[3]),protected_vs_b=paired(arms[2],arms[3]),
                          protected_vs_fixed=paired(arms[0],arms[3]),
                          newly_acquired=len(acquired),retained_after_B=len(acquired & successes(arms[2])),
                          retained_protected=len(acquired & successes(arms[3])))
    result['selection']=dict(a_scores=aw['scores'],a_index=aw['index'],b=bw)
    p.save('result.json',result)


if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    else:
        try:workflow()
        except BaseException:
            Path('failed'+os.environ['PART']).touch()
            raise
