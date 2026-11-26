import ast
import collections
import copy
import importlib.util
import json
import os
import random
import re
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import requests
import torch
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

ROOT = Path(os.environ['SOURCE']) / 'artifacts/source/c'
BASE = 'http://127.0.0.1:18476/fhir/'
DATA = ROOT / 'data/medagentbench'
SOURCE = ROOT / 'src/server/tasks/medagentbench'
MODEL = os.environ['MODEL']
SEED = 261126
families = ['task' + str(i) for i in range(1, 11)]
session = requests.Session()
session.trust_env = False


def save(name, data):
    tmp = Path(name + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    tmp.replace(name)


def family(c):
    return c['id'].split('_')[0]


def split(cases):
    rng = random.Random(SEED)
    patients = sorted({c['eval_MRN'] for c in cases if c.get('eval_MRN')})
    for _ in range(5000):
        rng.shuffle(patients)
        a, b = len(patients)//4, len(patients)//2
        out = {}
        for name, pool, n in [('discovery',patients[:a],2),('selection',patients[a:b],2),('later',patients[b:],4)]:
            part = []
            for f in families:
                options = [c for c in cases if family(c)==f and c.get('eval_MRN') in pool]
                part.extend(rng.sample(options, min(n,len(options))))
            out[name] = part
        if list(map(len,out.values())) == [20,20,40]:
            return out
    raise RuntimeError('could not form balanced groups')


def scorer():
    pkg = ModuleType('_ref')
    pkg.__path__ = [str(SOURCE)]
    sys.modules['_ref'] = pkg
    spec = importlib.util.spec_from_file_location('_ref.refsol', SOURCE/'refsol.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    def read(url):
        assert url.startswith(BASE)
        res = session.get(url, timeout=20)
        res.raise_for_status()
        return {'data':res.text, 'status_code':res.status_code}
    mod.send_get_request = read
    mod.print = lambda *a,**kw: None
    return mod


def grade(c, actions, answer, reads):
    if answer is None:
        return False, False
    hist = [SimpleNamespace(role=r,content=t) for r,t in actions]
    # A terminal action has no POST receipt. The upstream reader indexes i+1
    # even when the word POST occurs inside a final answer.
    if hist and hist[-1].role=='agent' and 'POST' in hist[-1].content:
        hist.append(SimpleNamespace(role='user',content=''))
    official = bool(getattr(ref,family(c))(c,SimpleNamespace(result=answer,history=hist),BASE))
    code = {'task5':'MG','task9':'K','task10':'A1C'}.get(family(c))
    relevant = any(path=='Observation' and str(c['eval_MRN']) in q.get('patient',[]) and code in q.get('code',[]) for path,q in reads)
    return official, official and (not code or relevant)


def audit(groups):
    c = next(c for c in groups['discovery'] if family(c)=='task3')
    p = dict(resourceType='Observation',category=[{'coding':[dict(system='http://hl7.org/fhir/observation-category',code='vital-signs',display='Vital Signs')]}], code={'text':'BP'},effectiveDateTime='2023-11-13T10:15:00+00:00',status='final',valueString='118/77 mmHg',subject={'reference':'Patient/'+c['eval_MRN']})
    wrong = dict(p,subject={'reference':'Patient/control'})
    missing = p.copy()
    del missing['status']
    def check(payloads):
        h = []
        for x in payloads:
            h.extend([('agent','POST '+BASE+'Observation\n'+json.dumps(x)),('user','POST request accepted')])
        return grade(c,h+[('agent','FINISH([])')],'[]',[])[0]
    result = {k:check(v) for k,v in [('valid',[p]),('empty',[]),('wrong_patient',[wrong]),('missing',[missing]),('duplicate',[p,p])]}
    assert list(result.values()) == [True,False,False,False,False], result
    cond = [c for c in groups['discovery'] if family(c) in ('task5','task9','task10')]
    vals = [grade(c,[('agent','FINISH([])')],'[]',[]) for c in cond]
    result.update(conditional_n=len(vals),conditional_empty_official=sum(v[0] for v in vals),conditional_empty_qualified=sum(v[1] for v in vals))
    save('audit.json', result)


def words(s):
    return set(re.findall(r'[a-z]+',re.sub(r'\bS\d+\b','',s).lower()))


def nearest(q, items):
    def rank(item):
        a,b = words(q),words(item['q'])
        return len(a&b)/max(1,len(a|b))
    return max(items,key=rank)


def clip(s,n):
    return tok.decode(tok.encode(s,add_special_tokens=False)[:n],skip_special_tokens=True)


@torch.inference_mode()
def generate(chats,sample=False,limit=384):
    texts = [tok.apply_chat_template(c,tokenize=False,add_generation_prompt=True) for c in chats]
    x = tok(texts,return_tensors='pt',padding=True).to('cuda')
    args = dict(do_sample=sample,max_new_tokens=limit,pad_token_id=tok.pad_token_id)
    if sample:
        args['temperature']=0.6
    y = model.generate(**x,**args)
    return tok.batch_decode(y[:,x.input_ids.shape[1]:],skip_special_tokens=True)


def fetch(url):
    p = urlsplit(url)
    route = p.path.removeprefix('/fhir/')
    if (p.scheme,p.netloc)!=('http','127.0.0.1:18476') or p.path!='/fhir/'+route or route not in get_routes:
        return 'Invalid local GET endpoint.', None
    try:
        r = session.get(url, params={'_format':'json'},timeout=15,allow_redirects=False)
        if r.status_code >= 500:
            raise RuntimeError(f'FHIR service {r.status_code}')
        if r.status_code != 200:
            return 'FHIR HTTP '+str(r.status_code), None
        data = r.json()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError('FHIR transport failure') from exc
    rows=[]
    for entry in data.get('entry',[]):
        x=entry['resource']
        rows.append({k:x[k] for k in ('resourceType','id','identifier','name','birthDate','subject','code','effectiveDateTime','valueQuantity','valueString') if k in x})
    rows.sort(key=lambda x:x.get('effectiveDateTime',''),reverse=True)
    total=data.get('total',len(rows))
    return json.dumps(dict(total=total,returned=len(rows[:24]),truncated=len(rows)>24 or total>len(rows),rows=rows[:24]),separators=(',',':')), (route,parse_qs(p.query))


def run(cases,mode,hints=None,sample=False):
    file=Path(mode+'.json')
    done=json.loads(file.read_text()) if file.exists() else []
    assert [r['id'] for r in done] == [c['id'] for c in cases[:len(done)]]
    for start in range(len(done),len(cases),4):
        torch.manual_seed(SEED + start + sum(map(ord, mode)))
        chunk=cases[start:start+4]
        batch=[]
        for c in chunk:
            hint=clip(hints(c),192) if hints else ''
            prompt=template.format(api_base=BASE.rstrip('/'),functions=docs,context=c['context'],question=c['instruction'])
            if hint:
                prompt+='\nPrior workflow memory (use only if relevant): '+hint
            batch.append(dict(chat=[dict(role='user',content=prompt)],hist=[],reads=[],answer=None,active=True,status='limit'))
        for _ in range(8):
            live=[b for b in batch if b['active']]
            if not live:
                break
            for b,raw in zip(live,generate([b['chat'] for b in live],sample)):
                text=raw.strip().replace('```tool_code','').replace('```','').strip()
                b['hist'].append(('agent',text))
                b['chat'].append(dict(role='assistant',content=text))
                if text.startswith('FINISH(') and text.endswith(')'):
                    b.update(active=False,answer=text[7:-1],status='finish')
                    continue
                if text.startswith('GET '):
                    resp,read=fetch(text[4:].strip())
                    if read: b['reads'].append(read)
                elif text.startswith('POST '):
                    try:
                        json.loads(text.partition('\n')[2])
                        resp='POST request accepted and executed successfully.'
                    except ValueError:
                        resp='Invalid POST request'
                else:
                    b.update(active=False,status='invalid')
                    continue
                b['hist'].append(('user',resp))
                b['chat'].append(dict(role='user',content=resp))
        for b,c in zip(batch,chunk):
            o,g=grade(c,b['hist'],b['answer'],b['reads'])
            done.append(dict(id=c['id'],q=c['instruction'],history=b['hist'],status=b['status'],official=o,qualified=g,steps=sum(r=='agent' for r,_ in b['hist'])))
        save(file.name,done)
        print(mode,len(done),'/',len(cases),flush=True)
    return done


def excerpt(row):
    return '\n'.join(t if r=='agent' else 'Response: '+clip(t,70) for r,t in row['history'])


def summarize(arms):
    base=arms['base']
    result={}
    for k,rows in arms.items():
        assert [r['id'] for r in base]==[r['id'] for r in rows]
        result[k]=dict(n=len(rows),official=sum(r['official'] for r in rows),qualified=sum(r['qualified'] for r in rows),gains=sum(r['qualified'] and not b['qualified'] for b,r in zip(base,rows)),losses=sum(b['qualified'] and not r['qualified'] for b,r in zip(base,rows)),mean_steps=sum(r['steps'] for r in rows)/len(rows),by_family={f:sum(r['qualified'] for r in rows if r['id'].split('_')[0]==f) for f in families},status=dict(collections.Counter(r['status'] for r in rows)))
    return result


if __name__=='__main__':
    cases=json.loads((DATA/'test_data_v2.json').read_text())
    groups=split(cases)
    if os.getenv('EXPAND_DISCOVERY')=='1':
        reserved={c['eval_MRN'] for name in ('selection','later') for c in groups[name]}
        original={c['id'] for c in groups['discovery']}
        extra=[c for c in cases if c.get('eval_MRN') and c['eval_MRN'] not in reserved and c['id'] not in original]
        groups['discovery'].extend(extra)
        assert len(groups['discovery']) == 160
    manifest={k:[c['id'] for c in v] for k,v in groups.items()}
    if Path('split.json').exists():
        assert json.loads(Path('split.json').read_text())==manifest
    save('split.json',manifest)
    ref=scorer()
    audit(groups)
    print('fixtures passed; loading model',flush=True)
    docs=json.dumps(json.loads((DATA/'funcs_v1.json').read_text()))
    get_routes={f['name'].split('/')[-1] for f in json.loads(docs) if f['name'].startswith('GET ')}
    docs=docs.replace('{api_base}',BASE.rstrip('/'))
    template=next(ast.literal_eval(n.value) for n in ast.parse((SOURCE/'__init__.py').read_text()).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='MedAgentBench_prompt')
    torch.manual_seed(SEED)
    tok=AutoTokenizer.from_pretrained(MODEL,padding_side='left',local_files_only=True)
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL,torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},local_files_only=True).eval()
    disc=run(groups['discovery'],'discover1')+run(groups['discovery'],'discover2',sample=True)
    bank=json.loads(Path('bank.json').read_text()) if Path('bank.json').exists() else []
    for f in ([] if bank else families):
        examples=[r for r in disc if r['id'].split('_')[0]==f]
        examples.sort(key=lambda r:r['qualified'], reverse=True)
        traces='\n'.join('Outcome '+str(r['qualified'])+'; '+clip(excerpt(r),480) for r in examples)
        prompt='Derive one reusable FHIR procedure from these scored development attempts. Describe when to read, how to use visible evidence and when to stop. No patient identifiers, concrete lab results or example-specific answers. Output only a concise instruction.\nTask: '+examples[0]['q']+'\n'+traces
        hint=generate([[dict(role='user',content=prompt)]],limit=192)[0]
        bank.append(dict(q=examples[0]['q'],hint=re.sub(r'\bS\d+\b','[patient]',hint)))
    save('bank.json',bank)
    skill=lambda c:nearest(c['instruction'],bank)['hint']
    def retrieve(c):
        # First match instruction, then prefer a successful trajectory of that type.
        match=nearest(c['instruction'],disc)
        bucket=[r for r in disc if r['id'].split('_')[0]==match['id'].split('_')[0]]
        return excerpt(nearest(c['instruction'],[r for r in bucket if r['qualified']] or bucket))
    val=run(groups['selection'],'valbase')
    edit=run(groups['selection'],'valskill',skill)
    scores=summarize({'base':val,'candidate':edit})
    d=scores['candidate']
    accept=d['gains']>d['losses'] and d['losses']<=1
    save('policy.json',dict(accept=accept,selection=scores))
    later=groups['later']
    b=run(later,'baseline')
    retry=run(later,'retry',sample=True)
    raw=run(later,'retrieval',retrieve)
    selected=run(later,'selected',skill) if accept else b
    summary=summarize(dict(base=b,retry=retry,retrieval=raw,selected=selected))
    save('aggregate.json',dict(selection=scores,accept=accept,later=summary,oracle_pass_at_2=sum(x['qualified'] or y['qualified'] for x,y in zip(b,retry))))
