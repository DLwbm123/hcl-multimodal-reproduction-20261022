import json,os,sys
from collections import Counter
from datetime import datetime,timedelta
from urllib.parse import urlsplit,parse_qs
import compare as v
import prior as p

CLOCK=datetime.fromisoformat('2023-11-13T10:15:00+00:00')
OPS=[(window,op) for window in (True,False) for op in ('latest','mean','min','max')]

def points(reply,mrn):
    obj=json.loads(reply)
    assert all(s['subject']=='Patient/'+mrn for s in obj['series'])
    out=sorted((t,x) for s in obj['series'] for t,x in s['measurements'])
    assert len(out)==obj['returned']
    return out

def visible(data,window):
    return [x for t,x in data if not window or CLOCK-timedelta(days=1)<=datetime.fromisoformat(t)<=CLOCK]

def apply(data,window,op):
    xs=visible(data,window)
    if not xs:return -1
    if op=='latest':return xs[-1]
    if op=='mean':return sum(xs)/len(xs)
    return min(xs) if op=='min' else max(xs)

def features(data):
    return len(visible(data,True)),len(data),apply(data,False,'latest'),apply(data,True,'mean')

def kind(c):
    return 'mean' if 'average' in c['instruction'].lower() else 'latest'

def matches(kind,x,y):
    return abs(x-y)<0.1 if kind=='mean' else x==y

def build(cases):
    v.p.ref=v.p.scorer()
    bank={'mean':[],'latest':[]};examples={'mean':[],'latest':[]}
    for c in sorted(cases,key=lambda c:{'build':0,'gate':1,'test':2}[c['stage']]):
        good,rows,posts,ans=v.d.canonical(c,return_trace=True)
        ref=sorted((r['effectiveDateTime'],r['valueQuantity']['value']) for r in rows)
        data=[tuple(x) for x in c['points']]
        assert good and not posts and ref==data
        if c['stage']=='test':continue
        k=kind(c);examples[k].append((c['stage'],data,ans[0]))
        if c['stage']=='build':
            url=v.d.query(c);reply,_=v.d.model_fetch(url)
            assert points(reply,c['eval_MRN'])==data
            text='GET '+url+'\nTool returned: '+reply+'\nFINISH('+json.dumps(ans)+')'
            bank[k].append(dict(features=features(data),demo=text.replace(c['eval_MRN'],'<example patient>')))
    policy={}
    for k,ex in examples.items():
        train=[(d,y) for stage,d,y in ex if stage=='build']
        gate=[(d,y) for stage,d,y in ex if stage=='gate']
        op=max(OPS,key=lambda op:sum(matches(k,apply(d,*op),y) for d,y in train))
        n=sum(matches(k,apply(d,*op),y) for d,y in gate)
        policy[k]=dict(op=op,build=len(train),gate=len(gate),success=n,enabled=n==len(gate) and n>0)
    v.p.save('bank.json',bank);v.p.save('policy.json',policy)
    v.p.save('controls.json',dict(checked=len(cases)))

def infer(test):
    v.setup_model()
    bank=p.read('bank');raw=v.p.generate
    def demo(chats,*args,**kwargs):
        decorated=[]
        for chat in chats:
            action,reply=chat[1]['content'],chat[2]['content']
            mrn=parse_qs(urlsplit(action[4:]).query)['patient'][0]
            query='average cbg of the patient' in chat[0]['content'].lower()
            f=features(points(reply,mrn))
            def distance(ex):
                a,b,c,d=ex['features']
                return abs(a-f[0])+abs(b-f[1])+(abs(c-f[2])+abs(d-f[3]))/100
            ex=min(bank['mean' if query else 'latest'],key=distance)
            assert query==('what is the average cbg' in chat[0]['content'].lower())
            chat=chat[:2]+[dict(role='user',content=reply+'\nVerified full demonstration from another patient. Adapt to the current findings.\n'+ex['demo'])]+chat[3:]
            seq=v.p.tok.apply_chat_template(chat,tokenize=True,add_generation_prompt=True)
            lim=v.p.model.config.text_config.max_position_embeddings if hasattr(v.p.model.config,'text_config') else v.p.model.config.max_position_embeddings
            assert len(seq)+384<=lim
            decorated.append(chat)
        return raw(decorated,*args,**kwargs)
    i=int(os.environ['PART'])
    for arm in ('base','demo'):
        v.p.generate=raw if arm=='base' else demo
        v.p.run(test[i::4],arm+str(i),initial_query=v.d.query,batch_size=2)

def report(test):
    v.p.ref=v.p.scorer()
    policy=p.read('policy');arms={}
    for arm in ('base','demo'):
        rs=[r for i in range(4) for r in p.read(arm+str(i))]
        arms[arm]={r['id']:r for r in rs}
        assert len(rs)==len(arms[arm])==len(test)
    stats={}
    for c in test:
        k=kind(c);a,b=[arms[t][c['id']] for t in arms]
        assert a['history'][:2]==b['history'][:2]
        h=a['history'][:2];data=points(h[1][1],c['eval_MRN'])
        assert data==[tuple(x) for x in c['points']]
        def grade(x):
            ans=json.dumps([x]);hist=h+[('agent','FINISH('+ans+')')]
            return v.p.grade(c,hist,ans,[('Observation',parse_qs(urlsplit(h[0][1][4:]).query))])[1]
        window='24 hours' in (c['instruction']+' '+c['context']).lower()
        scores=dict(base=a['qualified'],demo=b['qualified'],text=grade(apply(data,window,k)),fitted=grade(apply(data,*policy[k]['op'])) if policy[k]['enabled'] else a['qualified'])
        for group in ('all',k,k+'_'+c['group']):
            s=stats.setdefault(group,Counter());s['n']+=1
            for target,hits in scores.items():s[target]+=hits
            for target,row in [('base',a),('demo',b)]:
                s[target+'_model_turns']+=row['steps']-1
                s[target+'_tool_calls']+=sum(role=='agent' and t.startswith(('GET ','POST ')) for role,t in row['history'])
            for target,active in [('text',True),('fitted',policy[k]['enabled'])]:
                s[target+'_model_turns']+=0 if active else a['steps']-1
                s[target+'_tool_calls']+=1 if active else sum(role=='agent' and t.startswith(('GET ','POST ')) for role,t in a['history'])
            for arm in ('demo','text','base'):
                x,y=scores[arm],scores['fitted']
                s[arm+'_gain']+=y and not x;s[arm+'_loss']+=x and not y;s[arm+'_retained']+=x and y
    v.p.save('result.json',dict(policy=policy,patients=p.read('counts'),results=stats))

if __name__=='__main__':
    cases=p.read('cases');test=[c for c in cases if c['stage']=='test']
    if sys.argv[1]=='build':build(cases)
    elif sys.argv[1]=='infer':
        assert p.read('controls')['checked']==len(cases)
        infer(test)
    else:report(test)
