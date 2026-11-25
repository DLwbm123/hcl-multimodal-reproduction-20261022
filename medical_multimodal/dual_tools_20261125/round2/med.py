import json
import re
from collections import Counter
from PIL import Image
from clinical import Backend, DATA, NAMES, actions, data, evaluate, split
from adapt import append, consolidate, metrics, retrieval
from shared import calls, example, models, save

SEED=('Choose the next action from the clinical question and evidence already returned. '
      'Identify the information still needed. Finish when the available evidence supports an answer.')
MENU='A whole-image description; B localized visual attributes; C OCR for visible text and measurements; D numeric calculator; E finish.'
CODES=dict(zip(NAMES,'ABCD'))


def main():
    cases,refs=data()
    fit,gate,test=split()
    choose,gen=models()
    backend=Backend()
    print('tools ready',flush=True)
    def context(k):
        c=cases[k]; path=DATA/c['images'][0]
        with Image.open(path) as src: img=src.convert('RGB')
        return c['question'],img,path

    def visible(q,notes):
        return 'Question: '+q+'\nObserved:\n'+('\n'.join(n['name']+': '+n['text'][:450] for n in notes) or '(none)')

    def available(notes):
        used={n['name'] for n in notes if n['ok']}
        return [c for n,c in CODES.items() if n not in ('ImageDescription','OCR') or n not in used]+(['E'] if notes else [])

    def call(q,img,path,notes,name):
        ctx=visible(q,notes)
        req={'name':name,'path':str(path)}
        if name=='RegionAttributeDescription':
            raw=gen(ctx+'\nSelect a relevant box to resolve the remaining visual question. Exactly four integer coordinates from 0 to 1000, x1 y1 x2 y2.',40,img)
            nums=re.findall(r'(?<![\w.])-?\d+(?![\w.])',raw)
            if len(nums)==4:
                x,y,X,Y=map(int,nums)
                box=[x*img.width//1000,y*img.height//1000,X*img.width//1000,Y*img.height//1000]
            if len(nums)!=4 or not (0<=x<X<=1000 and 0<=y<Y<=1000 and box[0]<box[2] and box[1]<box[3]):
                return dict(name=name,ok=False,text='Invalid box',arg=raw[:80])
            req.update(box=box,attribute='Visible features relevant to: '+q)
        elif name=='Calculator':
            req['expression']=gen(ctx+'\nReturn ONLY a numeric expression using the observed values and + - * / parentheses.',40).strip()
        arg=json.dumps({k:v for k,v in req.items() if k not in ['name','path']},sort_keys=True)
        if any(n['name']==name and n['arg']==arg for n in notes):
            return dict(name=name,ok=False,arg=arg,text='Identical request already attempted; no new evidence.')
        out=backend.call(req)
        return dict(name=name,ok=out['ok'],arg=arg,text=out['text'])

    def policy(q,img,notes,skill,bank):
        state=visible(q,notes)
        hint=skill
        if bank:
            stage=notes[-1]['name'] if notes else 'start'
            hint+='\nPrior training decision:\n'+retrieval(state,stage,bank)
        opts=available(notes)
        return choose(state,{'image':img},hint+'\n'+MENU+'\nAvailable letters: '+' '.join(opts),opts)

    def episode(k,skill,bank):
        q,img,path=context(k);notes=[]
        for _ in range(4):
            c=policy(q,img,notes,skill,bank)
            if c=='E':break
            notes.append(call(q,img,path,notes,NAMES[ord(c)-65]))
        answer=gen(visible(q,notes)+'\nGive the concise clinical answer supported by this evidence.',96)
        return dict(notes=notes,answer=answer)

    def outcome(k,row):
        exp=actions(refs[k])
        seq=[n['name'] for n in row['notes'] if n['ok']]
        judged=choose('Reference acceptance criteria: '+json.dumps(refs[k]['gt_answer'])+'\nAnswer: '+row['answer'],{},
                  'A = answer satisfies the reference clinically including key details, B = incorrect or incomplete. Return A or B.',list('AB'))
        return dict(correct=int(judged=='A'),f1=evaluate(seq,exp),exact=int(seq==exp),
                    steps=len(row['notes']),invalid=sum(not n['ok'] for n in row['notes']))

    try:
        bank=[]; examples=[]; edit_examples=[]
        for i,k in enumerate(fit,1):
            q,img,path=context(k); notes=[]
            ref=refs[k]['dialogs']
            targets=[(t['function']['name'],str(msg.get('thought',''))) for msg in ref for t in msg.get('tool_calls',[])]
            for step,(name,rationale) in enumerate(targets[:4]+[('finish','All reference evidence has been acquired.')]):
                if step>=4:break
                if name!='finish' and CODES[name] not in available(notes):continue
                state=visible(q,notes)
                c=policy(q,img,notes,SEED,None)
                predicted='finish' if c=='E' else NAMES[ord(c)-65]
                stage=notes[-1]['name'] if notes else 'start'
                row=dict(visible=state,stage=stage,next=name)
                # Fit-only teacher explanations and actual returned evidence never enter evaluation input.
                response=call(q,img,path,notes,name) if name!='finish' else None
                detail=state+'\nReference rationale for this transition: '+rationale[:250]
                if response:detail+='\nWhat the executed next tool returned: '+response['text'][:220]
                examples.append(example(k+'.'+str(step),detail,predicted,name))
                edit_examples.append(dict(stage=stage,prior_result=(notes[-1]['text'][:180] if notes else '(none)'),
                    question=q[:150],wrong_or_chosen=predicted,next=name,reason=rationale[:200],
                    next_result=response['text'][:160] if response else 'finish'))
                bank.append(row)
                if response:notes.append(response)
            if i%5==0:print('fit',i,flush=True)
        memory=consolidate(examples,gen)
        # Cover transition types and mistakes before drawing a second example per type.
        groups={}
        for x in edit_examples:groups.setdefault((x['stage'],x['next']),[]).append(x)
        for v in groups.values():v.sort(key=lambda x:x['wrong_or_chosen']==x['next'])
        evidence=[v[j] for j in range(2) for v in groups.values() if len(v)>j][:24]
        instruction=gen('Write one short transferable procedure for choosing the next clinical evidence tool. '
            'Specify observable triggers for reading text, inspecting a specific region, arithmetic, and stopping. '
            'Use these fit transitions and observed results. Output only the procedure, no diagnoses or case details.\n'
            +json.dumps(evidence)+'\nHCL summaries: '+json.dumps(memory['abstract']),180).strip()
        skill=append(SEED,instruction)
        save('private_learning',dict(skill=skill,memory=memory,bank=bank))
        result={}; accepted=False
        for phase,ids in [('gate',gate),('test',test)]:
            rows={};scores=[]
            for i,k in enumerate(ids,1):
                base=episode(k,SEED,None); cand=episode(k,skill,bank)
                a,b=outcome(k,base),outcome(k,cand)
                rows[k]=dict(baseline=base,candidate=cand,base_score=a,candidate_score=b)
                scores.append((a,b))
                if i%5==0: print(phase,i,flush=True)
            left,right=map(lambda side:metrics(list(side)),zip(*scores))
            if phase=='gate': accepted=right['f1']>left['f1'] and right['correct']>=left['correct']
            result[phase]=dict(n=len(ids),baseline=left,candidate=right,deployed=right if accepted else left,
                gain=sum(b['correct']>a['correct'] for a,b in scores),loss=sum(b['correct']<a['correct'] for a,b in scores),
                transitions=Counter(n['name'] for row in rows.values() for n in row['candidate']['notes']))
            save('private_'+phase,rows)
            result.update(accepted=accepted,tool_executions=dict(backend.actual),model_calls=dict(calls))
            save('result',result)
    finally: backend.close()
