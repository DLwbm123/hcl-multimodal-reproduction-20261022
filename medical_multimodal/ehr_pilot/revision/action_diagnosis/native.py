import json,os
from urllib.parse import parse_qs,urlsplit
import torch
import interface as b
from interface import a,o,v,p,d


@torch.inference_mode()
def complete(chat,prefix):
    text=p.tok.apply_chat_template(chat,tokenize=False,add_generation_prompt=True)+prefix
    x=p.tok(text,return_tensors='pt',add_special_tokens=False).to(p.model.device)
    budget=383-len(p.tok.encode(prefix,add_special_tokens=False))
    y=p.model.generate(**x,do_sample=False,max_new_tokens=budget,pad_token_id=p.tok.pad_token_id)
    return prefix+p.tok.decode(y[0,x.input_ids.shape[1]:],skip_special_tokens=True).strip()


def simulate(c,obs,card,plan,truth,ans,oracle):
    text=p.template.format(api_base=p.BASE.rstrip('/'),functions=p.docs,context=c['context'],question=c['instruction'])
    text+='\nPrior workflow memory (use only if relevant): '+card
    hist=[('agent','GET '+obs['url']),('user',obs['reply'])]
    chat=[dict(role='user',content=text),dict(role='assistant',content=hist[0][1]),dict(role='user',content=obs['reply']+'\n'+o.hint(plan))]
    reads=[('Observation',parse_qs(urlsplit(obs['url']).query))]
    old=d.fetch
    try:
        d.fetch=lambda u:(json.dumps({'rows':obs['rows']}),reads[0])
        ok,_,targets,_=d.canonical(c,return_trace=True)
        assert ok
    finally:d.fetch=old
    actions=[];payloads=[];raw=[];answer=None
    for _ in range(7):
        kind=(truth[len(payloads)] if len(payloads)<len(truth)-1 else 'FINISH') if oracle else b.choose(chat)
        actions.append(kind)
        prefix='FINISH([' if kind=='FINISH' else 'POST '+p.BASE+kind+'\n{'
        out=complete(chat,prefix)
        raw.append(out);hist.append(('agent',out));chat.append(dict(role='assistant',content=out))
        if kind=='FINISH':
            if out.endswith(')'):answer=out[7:-1]
            break
        try:
            body=json.loads(out.partition('\n')[2])
            if not isinstance(body,dict):raise ValueError()
            payloads.append((kind,body));reply='POST request accepted and executed successfully.'
        except ValueError:reply='Invalid POST request'
        hist.append(('user',reply));chat.append(dict(role='user',content=reply))
    official,hit=p.grade(c,hist,answer,reads)
    try:terminal=json.loads(answer)==ans if answer is not None else False
    except ValueError:terminal=False
    errors=set()
    for reference,(_,obj) in zip(targets,payloads):
        src=b.flat(obj);errors.update(k for k,value in b.flat(reference).items() if src.get(k)!=value)
    return dict(hit=hit,official=official,terminal=terminal,attempted=any(k!='FINISH' for k in actions),accepted=bool(payloads),
                resource_sequence=[x[0] for x in payloads]==list(truth[:-1]),first=actions[0],actions=actions,raw=raw,field_errors=sorted(errors))


def main():
    v.setup_model()
    j=int(os.environ['PART'])
    rows=[]
    for c,obs,card,plan,_ in list(a.cases())[j::4]:
        truth,ans=o.reference(c,obs)
        if len(truth)>1:
            row={'id':c['id']}
            for arm in ('free','forced'):row[arm]=simulate(c,obs,card,plan,truth,ans,arm=='forced')
            rows.append(row);p.save('c'+str(j)+'.json',rows)
    p.save('c'+str(j)+'.json',rows)

if __name__=='__main__':
    import sys,collections
    if sys.argv[1]=='run':main()
    else:
        rows=[r for j in range(4) for r in o.read('c'+str(j))]
        assert len(rows)==7
        out={}
        for arm in ('free','forced'):
            rr=[r[arm] for r in rows]
            out[arm]={k:sum(r[k] for r in rr) for k in ('hit','official','terminal','attempted','accepted','resource_sequence')}
            out[arm]['fields']=dict(collections.Counter(k for r in rr for k in r['field_errors']))
            out[arm]['first']=dict(collections.Counter(r['first'] for r in rr))
        out['pair']=dict(gain=sum(r['forced']['hit'] and not r['free']['hit'] for r in rows),loss=sum(r['free']['hit'] and not r['forced']['hit'] for r in rows))
        p.save('native.json',out)
