import json
import os
import random
import re
import subprocess
from collections import Counter
from pathlib import Path
from PIL import Image
from shared import ROOT, calls, example, guide, learn, models, save

BASE=Path(os.environ['CTA_ROOT'])
DATA=Path(os.environ['DATA_ROOT'])
NAMES=['ImageDescription','RegionAttributeDescription','OCR','Calculator']
SEED='Choose the evidence tool that best resolves the question given the evidence already obtained. Finish once sufficient.'
MENU=('A ImageDescription (whole-image report), B RegionAttributeDescription (inspect a localized region), '
      'C OCR (read visible text), D Calculator (compute from observed numbers), E finish.')
def data():
    refs=json.loads((BASE/'data/dataset.json').read_text())
    cases={str(r['case_id']):r for r in map(json.loads,(DATA/'tasks.jsonl').read_text().splitlines())}
    assert set(cases)==set(refs)
    return cases,refs


def actions(row):
    return [t['function']['name'] for msg in row['dialogs'] for t in msg.get('tool_calls',[])]


def split():
    cases,refs=data()
    keys=sorted(k for k,r in refs.items() if 'GoogleSearch' not in actions(r))
    assert len({tuple(cases[k]['images']) for k in keys})==len(keys)
    random.Random(20261125).shuffle(keys)
    return [keys[:30],keys[30:46],keys[46:]]


class Backend:
    def __init__(self):
        env=os.environ.copy()
        setup=json.loads((BASE/'state/runtime_environment.json').read_text())
        for k in ('EASYOCR_MODULE_PATH','HF_HOME','HF_HUB_CACHE','TRANSFORMERS_CACHE','TORCH_HOME','MMENGINE_HOME','HOME'):
            env[k]=setup[k]
        env.update(PYTHONPATH=str(BASE/'code/agentlego'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                   ENTRY=str(ROOT/'tool_server.py'),TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',LOCAL_ONLY='1')
        self.log=(ROOT/'backend.log').open('a')
        tool_env=Path(json.loads((BASE/'plan.json').read_text())['environment'])
        env['PATH']=str(tool_env/'bin')+os.pathsep+env['PATH']
        self.proc=subprocess.Popen([str(tool_env/'bin/python3'),'-Bu','-c',
                                    'import os,runpy;runpy.run_path(os.environ["ENTRY"],run_name="__main__")'],
                                   stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,env=env,cwd=BASE)
        if self.proc.stdout.readline().strip()!='READY':
            raise RuntimeError('local tool backend failed to initialize; see backend.log')
        self.cache={}
        self.actual=Counter()

    def call(self,req):
        key=json.dumps(req,sort_keys=True)
        if key not in self.cache:
            self.proc.stdin.write(key+'\n');self.proc.stdin.flush()
            record=self.proc.stdout.readline()
            if not record:raise RuntimeError('tool process terminated; see backend.log')
            self.cache[key]=json.loads(record)
            self.actual[req['name']]+=1
        return self.cache[key]

    def close(self):
        self.proc.stdin.close()
        self.proc.wait(timeout=45)
        self.log.close()


def evaluate(route,expected):
    a,b=set(route),set(expected)
    return 2*len(a&b)/max(1,len(a)+len(b))


def main():
    cases,refs=data();fit,gate,test=split()
    choose,gen=models()
    backend=Backend()
    print('four live tools ready',flush=True)
    def run(k,skill,memory=None):
        c=cases[k]; q=c['question']
        path=DATA/c['images'][0]
        with Image.open(path) as src: img=src.convert('RGB')
        hint=guide(skill,memory,q) if memory else skill
        trail=[]; notes=[]
        for step in range(4):
            ctx=q+'\nPrevious results:\n'+'\n'.join(notes)
            opts=list('ABCDE' if step else 'ABCD')
            code=choose(ctx,{'image':img},hint+'\n'+MENU,opts)
            if code=='E':break
            name=NAMES[ord(code)-ord('A')]
            req=dict(name=name,path=str(path))
            valid=True
            if name=='RegionAttributeDescription':
                raw=gen(q+'\nSelect one relevant region. Output only four integers x1,y1,x2,y2 on a 0 to 1000 normalized image grid.',36,img)
                nums=re.findall(r'(?<![\w.])-?\d+(?![\w.])',raw)
                valid=len(nums)==4
                if valid:
                    x,y,X,Y=map(int,nums)
                    valid=0<=x<X<=1000 and 0<=y<Y<=1000
                    box=[int(x*img.width/1000),int(y*img.height/1000),int(X*img.width/1000),int(Y*img.height/1000)]
                    valid=valid and box[2]>box[0] and box[3]>box[1]
                    req.update(box=box,attribute='visible features relevant to this question: '+q)
            elif name=='Calculator':
                req['expression']=gen(ctx+'\nReturn ONLY a numerical arithmetic expression using + - * / and parentheses from the observed evidence.',40).strip()
            result=backend.call(req) if valid else dict(ok=False,text='Invalid region coordinates')
            trail.append(dict(action=name,ok=result['ok']))
            notes.append(name+': '+result['text'])
        answer=gen(q+'\nTool evidence:\n'+'\n'.join(notes)+'\nGive the concise final clinical answer supported by the evidence.',96)
        return dict(answer=answer,trace=trail,notes=notes)

    def score(k,p):
        seq=[t['action'] for t in p['trace'] if t['ok']]
        expected=actions(refs[k])
        # The scorer sees only reference criteria and the answer, with no policy identifier.
        cmp='Reference acceptance criteria: '+json.dumps(refs[k]['gt_answer'])+'\nAnswer: '+p['answer']
        correct=choose(cmp,{},'A = answer satisfies the reference clinically including key details, B = incorrect or incomplete. Return A or B.',list('AB'))=='A'
        return dict(f1=evaluate(seq,expected),exact=int(seq==expected),correct=int(correct),
                    invalid=sum(not t['ok'] for t in p['trace']),steps=len(p['trace']))
    def means(rows):
        return {s:sum(r[s] for r in rows)/len(rows) for s in rows[0]}

    examples=[]; result={}
    previous=Path(os.environ['REUSE_BASELINE']) if os.environ.get('REUSE_BASELINE') else None
    try:
        if previous:
            assert json.loads((previous/'SUPERVISOR_EXIT.json').read_text())['success']
            examples=json.loads((previous/'private_fit.json').read_text())
            assert [r['task_id'] for r in examples]==fit
        else:
            for i,k in enumerate(fit,1):
                p=run(k,SEED)
                observed=' -> '.join(t['action'] for t in p['trace'] if t['ok'])
                expected=' -> '.join(actions(refs[k]))
                examples.append(example(k,cases[k]['question'],observed,expected))
                if i%5==0:print('fit',i,flush=True)
        save('private_fit',examples)
        skill,memory=learn(SEED,examples,gen)
        accepted=False
        for phase,ids in [('gate',gate),('test',test)]:
            scores=[];rows={}
            cache=json.loads((previous/('private_'+phase+'.json')).read_text()) if previous else {}
            if previous: assert set(cache)==set(ids)
            for i,k in enumerate(ids,1):
                first=cache[k][0] if previous else run(k,SEED)
                a=score(k,first)
                if skill and (phase=='gate' or accepted):
                    second=run(k,skill,memory); b=score(k,second)
                else:second,b=first,a
                rows[k]=[first,second];scores.append((a,b))
                if i%5==0:print(phase,i,flush=True)
            a,b=means([x[0] for x in scores]),means([x[1] for x in scores])
            if phase=='gate':accepted=bool(skill and b['f1']>a['f1'] and b['correct']>=a['correct'])
            save('private_'+phase,rows)
            result[phase]=dict(n=len(ids),baseline=a,learned=b,gain=sum(i['correct']<j['correct'] for i,j in scores),
                               loss=sum(i['correct']>j['correct'] for i,j in scores))
            result.update(accepted=accepted,valid_edit=bool(skill),reused_baseline=bool(previous),model_calls=dict(calls),tool_executions=dict(backend.actual))
            save('progress',result)
        save('result',result)
    finally:
        backend.close()
