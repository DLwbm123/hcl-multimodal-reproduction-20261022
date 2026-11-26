import collections
import json
import os
from pathlib import Path
from urllib.parse import parse_qs,urlsplit
import compare as v

def read(name): return json.loads(Path(name+'.json').read_text())

def prefix(c,row):
    hist=[]
    for role,s in row['history']:
        if role=='agent' and not s.startswith('GET '): return
        hist.append((role,s))
        if role!='user' or not s.startswith('{"returned":'):continue
        u=urlsplit(hist[-2][1][4:])
        q=parse_qs(u.query)
        if u.path=='/fhir/Observation' and c['eval_MRN'] in q.get('patient',[]) and 'MG' in q.get('code',[]):
            return hist,json.loads(s)

def parse(text):
    text=text.strip().replace('```tool_code','').replace('```','').strip()
    if text.startswith('POST '):return 'POST',text
    if text.startswith('FINISH(') and text.endswith(')'):return 'FINISH',text
    return 'other',text

def score(c,raw,write,bundle):
    action,text=parse(raw)
    kind='POST' if write else 'FINISH'
    if action!=kind:return False, False
    if write:
        try:json.loads(text.partition('\n')[2])
        except ValueError:return True,False
        hist=[('agent',text),('user','POST request accepted and executed successfully.'),('agent','FINISH([])')]
        ans='[]'
    else:
        hist=[('agent',text)]
        ans=text[7:-1]
    v.p.ref.send_get_request=lambda url:{'status_code':200,'data':json.dumps(bundle)}
    return True,v.p.grade(c,hist,ans,[('Observation',{'patient':[c['eval_MRN']],'code':['MG']})])[1]

def main():
    v.setup_model()
    cases=read('cases')
    truth=read('truth')
    training=next(c for c in cases if c['stage']=='train' and c['source']=='simulation' and v.d.kind(c)==5 and truth[c['id']])
    post=json.loads(json.dumps(truth[training['id']][0]))
    post['subject']['reference']='Patient/<current MRN>'
    post['authoredOn']='<time from current context>'
    for key in ['doseQuantity','rateQuantity']:post['dosageInstruction'][0]['doseAndRate'][0][key]['value']='<number for this tier>'
    edit=('Read the newest in-window numeric measurement. If replacement is required under the question rules, '
          'the next action must submit the current order: POST '+v.p.BASE+'MedicationRequest\n'+json.dumps(post,separators=(',',':'))+
          '\nFill the variable slots from this patient and task. After its receipt, or immediately when no order is required, use FINISH([]).')
    card=next(x['card'] for x in read('library') if x['key']==5)
    data={c['id']:c for c in cases}
    out=[];eligible=collections.Counter()
    for row in read('valid_baseline_5'):
        c=data[row['id']];ref=truth[c['id']];group=c['source']+('/write' if ref else '/no_write')
        eligible[group]+=1
        state=prefix(c,row)
        if state is None:continue
        history,payload=state
        obs=[{'resource':{'effectiveDateTime':time,'valueQuantity':{'value':val}}}
             for series in payload['series'] for time,val in series['measurements']]
        assert len(obs)==payload['returned']
        bundle={'entry':obs}
        target='POST '+v.p.BASE+'MedicationRequest\n'+json.dumps(ref[0]) if ref else 'FINISH([])'
        assert score(c,target,bool(ref),bundle)==(True,True)
        result={}
        for arm in ('original','early','late'):
            prompt=v.p.template.format(api_base=v.p.BASE.rstrip('/'),functions=v.p.docs,context=c['context'],question=c['instruction'])
            prompt+='\nPrior workflow memory (use only if relevant): '+card
            if arm=='early':prompt+='\n'+edit
            chat=[dict(role='user',content=prompt)]
            chat.extend(dict(role='assistant' if role=='agent' else 'user',content=s) for role,s in history)
            if arm=='late':chat[-1]['content']+='\nWorkflow instruction:\n'+edit
            raw=v.p.generate([chat])[0]
            typ,ok=score(c,raw,bool(ref),bundle)
            result[arm]=dict(type=typ,ok=ok,output=raw)
        out.append(dict(id=c['id'],group=group,arms=result))
        v.p.save('private.json',out)
        print(len(out),flush=True)
    totals={}
    for group,n in eligible.items():
        rows=[r['arms'] for r in out if r['group']==group]
        totals[group]=dict(available=n,measured=len(rows),
            correct={a:sum(r[a]['ok'] for r in rows) for a in ('original','early','late')},
            type_correct={a:sum(r[a]['type'] for r in rows) for a in ('original','early','late')},
            gains=sum(r['late']['ok'] and not r['early']['ok'] for r in rows),
            losses=sum(r['early']['ok'] and not r['late']['ok'] for r in rows))
    v.p.save('result.json',totals)
if __name__=='__main__':main()
