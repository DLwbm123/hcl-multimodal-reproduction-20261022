"""Paired historical-panel comparison with a prospectively frozen development gate."""
import ast
import collections
import json
import os
import random
import re
import sys
import time
from pathlib import Path

here=Path(__file__).resolve().parent
sys.path[:0]=[str(here.parent), str(here.parent/'calibration')]
import diagnose as d
import pilot as p

p.BASE=os.environ.get('FHIR_BASE','http://127.0.0.1:18478/fhir/')
N=int(os.environ.get('SHARDS','5'))
PRIOR=Path(os.environ['PRIOR'])
ROWS={c['id']:c for c in json.loads((p.DATA/'test_data_v2.json').read_text())}
SPLIT=json.loads((PRIOR/'split.json').read_text())


def split():
    dev=[ROWS[i] for i in SPLIT['discovery']]
    patients=sorted({c['eval_MRN'] for c in dev})
    rng=random.Random(261127)
    while True:
        rng.shuffle(patients)
        first=set(patients[:len(patients)//2])
        sets=[[c for c in dev if (c['eval_MRN'] in first)==flag] for flag in (True,False)]
        if all(len({d.kind(c) for c in s})==10 for s in sets):break
    hist=[ROWS[i] for i in SPLIT['later']]
    assert len(hist)==40 and not ({c['eval_MRN'] for c in hist}&set(patients))
    return *sets,hist


def prompt_question(c):
    text=c['instruction']
    text=re.sub(r'name .+? and DOB of [\d-]+','name <name> and DOB of <DOB>',text)
    text=re.sub(r'\bS\d+\b','<MRN>',text)
    text=re.sub(r'\d+/\d+ mmHg','<current BP>',text)
    return text


def material(c):
    good,rows,posts,ans=d.canonical(c,return_trace=True)
    assert good
    n=d.kind(c)
    steps=[]
    if n not in d.WRITES:
        if n==1:
            tail='Patient?given=<given>&family=<family>&birthdate=<DOB>'
            view='{"returned":"<count>","rows":[{"identifier":[{"value":"<matched MRN>"}],"name":"<matched name>","birthDate":"<DOB>"}]}'
        elif n==2:
            tail='Patient?identifier=<MRN>'
            view='{"returned":"<count>","rows":[{"identifier":[{"value":"<MRN>"}],"birthDate":"<birth date>"}]}'
        else:
            tail='Observation?patient=<MRN>&code='+d.CODES[n]
            view=('{"returned":"<count>","columns":["effectiveDateTime","value"],"series":[{"subject":"Patient/<MRN>","code":"<lab>","unit":"<unit>","measurements":[["<time>","<measurement>"], "... repeat for each real record"]}]}' if rows else '{"returned":0,"columns":["effectiveDateTime","value"],"series":[]}')
        steps.extend(['GET '+p.BASE+tail,'Tool returned (field view): '+view])
    for obj in posts:
        obj['subject']['reference']='Patient/<MRN>'
        for key in ('effectiveDateTime','authoredOn'):
            if key in obj:obj[key]='<now from context>'
        if 'valueString' in obj:obj['valueString']='<measured BP and unit>'
        if 'note' in obj:obj['note']['text']='<full referral text from question>'
        if 'occurrenceDateTime' in obj:obj['occurrenceDateTime']='<requested follow-up time>'
        if 'dosageInstruction' in obj:
            amount=obj['dosageInstruction'][0]['doseAndRate'][0]
            for k,v in amount.items():v['value']='<'+k+' from current lab and dosing instruction>'
        steps.extend(['POST '+p.BASE+obj['resourceType']+'\n'+json.dumps(obj,separators=(',',':')),'Tool: POST request accepted.'])
    result=({1:'["<matched MRN>"]',2:'[<age in full years>]',4:'[<latest in-window value>]',6:'[<mean of in-window values>]',7:'[<latest value>]',10:'[<latest value>,"<measurement timestamp>"]'}.get(n,'[]') if ans!=[-1] else '[-1]')
    steps.append('FINISH('+result+')')
    return dict(body='\n'.join(steps),write=bool(posts),informative=bool(ans and ans!=[-1]),rows=len(rows))


def setup_model():
    p.ref=p.scorer()
    p.fetch=d.model_fetch
    p.template=next(ast.literal_eval(n.value) for n in ast.parse((p.SOURCE/'__init__.py').read_text()).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='MedAgentBench_prompt')
    p.docs=json.dumps(d.DOCS).replace('{api_base}',p.BASE.rstrip('/'))
    p.tok=p.AutoTokenizer.from_pretrained(p.MODEL,padding_side='left',local_files_only=True)
    p.model=p.Qwen2_5_VLForConditionalGeneration.from_pretrained(p.MODEL,torch_dtype=p.torch.bfloat16,device_map={'':'cuda:0'},local_files_only=True).eval()
    p.model.lm_head.register_forward_pre_hook(lambda module,args:(args[0][:,-1:,:],))


def build():
    train,valid,hist=split()
    p.save('partition.json',dict(train=[c['id'] for c in train],valid=[c['id'] for c in valid],history=[c['id'] for c in hist]))
    out=[]
    for n in range(1,11):
        cases=[c for c in train if d.kind(c)==n]
        options=[(c,material(c)) for c in cases]
        chosen=[]
        for write in sorted({v['write'] for c,v in options}):
            chosen.append(min(((c,v) for c,v in options if v['write']==write),key=lambda pair:(not pair[1]['informative'],pair[1]['rows'],pair[0]['id'])))
        c=chosen[0][0]
        header='Completed examples from different patients, with variable slots. Execute a fresh workflow for the current question. Copy no receipt or old answer.\n'
        demo=header+'\n\n'.join(v['body'] for _,v in chosen)
        assert len(p.tok.encode(demo,add_special_tokens=False))<=1024
        rules=c['context'].replace(d.MONITOR.isoformat(),'<current time>')
        prompt=('Automatically extract a concise executable workflow card from these verified parameterized traces. Include when to apply, current-patient variable binding, exact tool syntax, arrays versus objects, result interpretation, branching and when to FINISH. Do not report any example-specific answer or claim any new action has happened. Use only the supplied documentation and examples.\nTask: '+prompt_question(c)+'\nContext: '+rules+'\nDocumented tools: '+p.docs+'\nExamples:\n'+demo)
        card=p.generate([[dict(role='user',content=prompt)]],limit=512)[0].strip()
        assert card and not re.search(r'\bS\d+\b',card)
        out.append(dict(key=n,q=prompt_question(c),sources=[x['id'] for x,_ in chosen],demo=demo,card=card,demo_tokens=len(p.tok.encode(demo,add_special_tokens=False)),card_tokens=len(p.tok.encode(card,add_special_tokens=False))))
        print('verified and distilled',n,flush=True)
    p.save('library.json',out)


def match(case,bank):
    return p.nearest(case['instruction'],bank)


def compare(arm,stage,idx):
    bank=json.loads(Path('library.json').read_text())
    ids=json.loads(Path('partition.json').read_text())[stage]
    cases=[ROWS[i] for i in ids][idx::N]
    hint=None if arm=='plain' else lambda c:match(c,bank)[arm]
    return p.run(cases,f'{stage}_{arm}_{idx}',hint,batch_size=2,hint_budget=1024)


def merged(arm,stage):
    ids=json.loads(Path('partition.json').read_text())[stage]
    data={}
    for ix in range(N):
        for row in json.loads(Path(f'{stage}_{arm}_{ix}.json').read_text()):
            assert row['id'] not in data
            data[row['id']]=row
    assert set(data)==set(ids)
    return [data[i] for i in ids]


def gate():
    b,c=merged('plain','valid'),merged('card','valid')
    bank=json.loads(Path('library.json').read_text())
    scores=collections.defaultdict(collections.Counter)
    for x,y in zip(b,c):
        assert x['id']==y['id']
        g=scores[match(ROWS[x['id']],bank)['key']]
        g.update(n=1,wins=int(y['qualified'] and not x['qualified']),losses=int(x['qualified'] and not y['qualified']))
    p.save('gate.json',dict(enabled=sorted(k for k,s in scores.items() if s['wins'] and not s['losses']),counts=scores))


def report():
    arms={k:merged(k,'history') for k in ('plain','demo','card')}
    bank=json.loads(Path('library.json').read_text())
    enabled=json.loads(Path('gate.json').read_text())['enabled']
    keys=[match(ROWS[r['id']],bank)['key'] for r in arms['plain']]
    arms['selected']=[row if k in enabled else b for row,b,k in zip(arms['card'],arms['plain'],keys)]
    historical={r['id'] for r in json.loads((PRIOR/'baseline.json').read_text()) if r['qualified']}
    def pair(ref,method):
        aa={r['id'] for r in ref if r['qualified']}
        bb={r['id'] for r in method if r['qualified']}
        return dict(correct=len(bb),gains=len(bb-aa),losses=len(aa-bb),retained=len(aa&bb))
    by={k:pair(arms['plain'],v) for k,v in arms.items()}
    for name,v in arms.items():
        by[name].update(official=sum(r['official'] for r in v),historical_retained=sum(r['qualified'] and r['id'] in historical for r in v),by_family=dict(collections.Counter(p.family(r) for r in v if r['qualified'])),model_steps=sum(r['steps'] for r in v),tools=sum(sum(w=='agent' and t.startswith(('GET ','POST ')) for w,t in r['history']) for r in v))
        by[name]['memory_tokens']=sum((0 if name=='plain' or (name=='selected' and k not in enabled) else next(item[('demo' if name=='demo' else 'card')+'_tokens'] for item in bank if item['key']==k)) for k in keys)
    p.save('result.json',dict(n=len(keys),historical_correct=len(historical),enabled=enabled,active=sum(k in enabled for k in keys),arms=by,selected_vs_demo=pair(arms['demo'],arms['selected']),teacher_examples=sum(len(item['sources']) for item in bank),partition={k:len(v) for k,v in json.loads(Path('partition.json').read_text()).items()}))


def wait_for(path):
    while not Path(path).exists():
        time.sleep(2)


if __name__=='__main__':
    idx=int(os.environ['PART'])
    setup_model()
    if idx==0 and not Path('library.json').exists():build()
    wait_for('library.json')
    for arm in ('plain','card'):compare(arm,'valid',idx)
    Path(f'done_valid_{idx}').touch()
    if idx==0:
        for j in range(N):wait_for(f'done_valid_{j}')
        gate()
    wait_for('gate.json')
    for arm in ('plain','demo','card'):compare(arm,'history',idx)
    Path(f'done_history_{idx}').touch()
    if idx==0:
        for j in range(N):wait_for(f'done_history_{j}')
        report()
