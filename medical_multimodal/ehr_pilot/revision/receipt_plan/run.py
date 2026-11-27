import collections
import datetime as dt
import json
import os
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import compare as v

p, d = v.p, v.d
p.BASE = 'http://127.0.0.1:18476/fhir/'
TYPES = {4,5,6,7,9,10}


def read(path):
    return json.loads(Path(path+'.json').read_text())


def same_query(url, wanted, extra=()):
    query, ref = urlsplit(url), urlsplit(wanted)
    fields = parse_qs(query.query,keep_blank_values=True)
    for key in extra:
        fields.pop(key,None)
    return (query.scheme,query.netloc,query.path,fields)==(ref.scheme,ref.netloc,ref.path,parse_qs(ref.query))


def source():
    partition=read('archive/partition')
    records={r['id']:r for r in read('inputs')}
    parts={k:[i for i in partition[k] if d.kind(v.ROWS[i]) in TYPES] for k in ('train','valid')}
    assert set(v.ROWS[i]['eval_MRN'] for i in parts['train']).isdisjoint(v.ROWS[i]['eval_MRN'] for i in parts['valid'])
    return parts,records


def observation(c,record):
    action,reply=record['history'][:2]
    url=d.query(c)
    assert action==['agent','GET '+url] and reply[0]=='user'
    data=json.loads(reply[1])
    rows=[]
    for series in data['series']:
        assert series['subject']=='Patient/'+c['eval_MRN'] and d.CODES[d.kind(c)] in {coding['code'] for coding in series['code']['coding']}
        for stamp,val in series['measurements']:
            rows.append(dict(effectiveDateTime=stamp,valueQuantity=dict(value=val)))
    assert len(rows)==data['returned']
    times=[(dt.datetime.fromisoformat(r['effectiveDateTime']),r['valueQuantity']['value']) for r in rows]
    when,val=max(times) if times else (None,0)
    hours=(d.MONITOR-when).total_seconds()/3600 if when is not None else 0
    recent=sum(d.MONITOR-dt.timedelta(hours=24)<=t<=d.MONITOR for t,_ in times)
    return dict(url=url,reply=reply[1],rows=rows,features=[len(rows),recent,hours,val])


def reference(c,obs):
    url=obs['url']
    def raw(query):
        assert same_query(query,url,('_count','_format'))
        return dict(status_code=200,data=json.dumps(dict(entry=[{'resource':r} for r in obs['rows']])))
    p.ref.send_get_request=raw
    old_fetch=d.fetch
    def fetch(query):
        assert same_query(query,url)
        return json.dumps({'rows':obs['rows']}),('Observation',parse_qs(urlsplit(url).query))
    try:
        d.fetch=fetch
        good,_,posts,answer=d.canonical(c,return_trace=True)
        assert good
    finally:
        d.fetch=old_fetch
    if not answer:
        form='empty list'
    elif answer==[-1]:
        form='one-element list with the missing-value sentinel'
    elif len(answer)==2:
        form='list with current numeric value followed by its timestamp'
    else:
        assert len(answer)==1
        form='one-element list with the requested numeric result'
    return tuple([obj['resourceType'] for obj in posts]+[form]),answer


def majority(rows):
    counts=collections.Counter(t for _,t,_ in rows)
    return min(counts,key=lambda t:(-counts[t],t))


def train(rows):
    mode=majority(rows)
    best=dict(mode=mode)
    score=sum(label==mode for _,label,_ in rows)
    for col in range(4):
        values=sorted({feat[col] for feat,_,_ in rows})
        for lo,hi in zip(values,values[1:]):
            cut=(lo+hi)/2
            left=[r for r in rows if r[0][col]<=cut]
            right=[r for r in rows if r[0][col]>cut]
            a,b=majority(left),majority(right)
            hits=sum(t==a for _,t,_ in left)+sum(t==b for _,t,_ in right)
            if hits>score:
                best.update(col=col,cut=cut,left=a,right=b)
                score=hits
    return best


def predict(rule,feat):
    if 'col' not in rule:
        return tuple(rule['mode'])
    return tuple(rule['left' if feat[rule['col']]<=rule['cut'] else 'right'])


def fit():
    p.ref=p.scorer()
    parts,records=source()
    bank=read('archive/library')
    groups=collections.defaultdict(list)
    for i in parts['train']:
        c=v.ROWS[i]
        state=observation(c,records[i])
        plan,_=reference(c,state)
        key=v.match(c,bank)['key']
        groups[key].append((state['features'],plan,c['eval_MRN']))
    rules={}
    for key,rows in groups.items():
        gain=loss=0
        for person in sorted({person for _,_,person in rows}):
            rule=train([r for r in rows if r[2]!=person])
            for feat,plan,subject in rows:
                if subject!=person:
                    continue
                fixed=rule['mode']==plan
                routed=predict(rule,feat)==plan
                gain+=routed and not fixed
                loss+=fixed and not routed
        rules[key]=dict(rule=train(rows),enabled=gain>0 and loss==0,loo_gain=gain,loo_loss=loss,n=len(rows))
    assert len(parts['train'])==50 and len(parts['valid'])==49
    p.save('rules.json',rules)


def hint(plan):
    writes=' '.join('POST '+kind+' then read its receipt.' for kind in plan[:-1])
    return 'Workflow continuation from verified training examples. '+writes+' Use FINISH with a '+plan[-1]+'. Bind every field to the current question and receipt.'


def simulate(c,obs,card,plan,truth_plan,truth_answer):
    prompt=p.template.format(api_base=p.BASE.rstrip('/'),functions=p.docs,context=c['context'],question=c['instruction'])
    prompt+='\nPrior workflow memory (use only if relevant): '+card
    history=[('agent','GET '+obs['url']),('user',obs['reply'])]
    chat=[dict(role='user',content=prompt),dict(role='assistant',content=history[0][1]),
          dict(role='user',content=obs['reply']+('\n'+hint(plan) if plan is not None else ''))]
    reads=[('Observation',parse_qs(urlsplit(obs['url']).query))]
    answer=None
    misses=0
    posted=[]
    for _ in range(7):
        action=p.generate([chat],limit=384)[0].strip().replace('```tool_code','').replace('```','').strip()
        history.append(('agent',action))
        chat.append(dict(role='assistant',content=action))
        if action.startswith('FINISH(') and action.endswith(')'):
            answer=action[7:-1]
            break
        if action.startswith('POST '):
            try:
                json.loads(action.partition('\n')[2])
                reply='POST request accepted and executed successfully.'
                posted.append(action.partition('\n')[0][5:].split('/')[-1])
            except ValueError:
                reply='Invalid POST request'
        elif action.startswith('GET '):
            if same_query(action[4:].strip(),obs['url']):
                reply=obs['reply']
            else:
                misses+=1
                reply='Requested query is unavailable in this recorded observation session.'
        else:
            break
        history.append(('user',reply))
        chat.append(dict(role='user',content=reply))
    official,hit=p.grade(c,history,answer,reads)
    try:
        actual=json.loads(answer) if answer is not None else None
    except ValueError:
        actual=None
    exact=actual==truth_answer
    if d.kind(c)==6 and isinstance(actual,list) and len(actual)==1 and type(actual[0]) in (int,float):
        exact=abs(actual[0]-truth_answer[0])<0.1
    return dict(hit=hit,official=official,full=hit and exact,terminal=exact,post_seq=tuple(posted)==truth_plan[:-1],
                unexpected_write=bool(posted) and len(truth_plan)==1,misses=misses,
                calls=sum(r=='agent' and t.startswith(('GET ','POST ')) for r,t in history),
                steps=sum(r=='agent' for r,_ in history))


def evaluate():
    v.setup_model()
    parts,records=source()
    bank=read('archive/library')
    rules=read('rules')
    shard=int(os.environ['PART'])
    ids=parts['valid'][shard::4]
    path='part'+str(shard)
    saved=read(path) if Path(path+'.json').exists() else []
    assert [r['id'] for r in saved]==ids[:len(saved)]
    for i in ids[len(saved):]:
        c=v.ROWS[i]
        obs=observation(c,records[i])
        item=v.match(c,bank)
        rule=rules[str(item['key'])]
        static=tuple(rule['rule']['mode'])
        routed=predict(rule['rule'],obs['features'])
        truth,ans=reference(c,obs)
        results={}
        for plan in (None,static,routed):
            if plan not in results:
                results[plan]=simulate(c,obs,item['card'],plan,truth,ans)
        selected=routed if rule['enabled'] else static
        saved.append(dict(id=i,write=len(truth)>1,changed=static!=routed,enabled=rule['enabled'],
                          plan_hits={a:t==truth for a,t in [('static',static),('routed',routed),('selected',selected)]},
                          arms={a:results[t] for a,t in [('card',None),('static',static),('routed',routed),('selected',selected)]}))
        p.save(path+'.json',saved)
        print(shard,len(saved),'/',len(ids),flush=True)


def report():
    rows=[r for i in range(4) for r in read('part'+str(i))]
    assert {r['id'] for r in rows}==set(source()[0]['valid']) and len(rows)==49
    groups=collections.defaultdict(list)
    for r in rows:
        for key in ['all',p.family(v.ROWS[r['id']]),'write' if r['write'] else 'no_write','changed' if r['changed'] else 'unchanged']:
            groups[key].append(r)
    stats={}
    for key,rr in groups.items():
        s=dict(n=len(rr),people=len({v.ROWS[r['id']]['eval_MRN'] for r in rr}),changed=sum(r['changed'] for r in rr))
        for arm in ('card','static','routed','selected'):
            s[arm]={k:sum(r['arms'][arm][k] for r in rr) for k in rr[0]['arms'][arm]}
            if arm!='card':
                s[arm]['plan_hits']=sum(r['plan_hits'][arm] for r in rr)
        for a,b in [('card','static'),('static','routed'),('static','selected'),('card','selected')]:
            pairs={}
            for metric in ('hit','full'):
                xy=[(r['arms'][a][metric],r['arms'][b][metric]) for r in rr]
                patients=collections.Counter()
                for r,(x,y) in zip(rr,xy):
                    patients[v.ROWS[r['id']]['eval_MRN']]+=int(y)-int(x)
                pairs[metric]=dict(gain=sum(y and not x for x,y in xy),loss=sum(x and not y for x,y in xy),
                                   retained=sum(x and y for x,y in xy),patients=dict(collections.Counter(
                                       'gain' if d>0 else 'loss' if d<0 else 'tie' for d in patients.values())))
            s[b+'_vs_'+a]=pairs
        stats[key]=s
    gate={k:{f:v[f] for f in ('n','loo_gain','loo_loss','enabled')} for k,v in read('rules').items()}
    p.save('result.json',dict(gate=gate,stats=stats))


if __name__=='__main__':
    {'fit':fit,'eval':evaluate,'report':report}[sys.argv[1]]()
