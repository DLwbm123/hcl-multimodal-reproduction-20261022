import json
from collections import Counter, defaultdict
from adapt import append
from previous_learning import update_memory, words
from shared import ROOT, example, save

TRUST='Adopt the revised answer when the new observation improves the support for its answer to the question; otherwise retain the prior answer.'
STOP='Finish if the accepted evidence resolves the question; continue if another available tool can resolve a specific remaining uncertainty.'


def outcome(a,b):
    return {(0,1):'corrected',(1,0):'spoiled',(1,1):'stayed_correct',(0,0):'stayed_wrong'}[a,b]


def future(now,later):
    if not later:
        return 'unobserved'
    if now:
        return 'already_correct'
    return 'later_corrected' if any(later) else 'no_observed_rescue'


def build_memory(rows,gen,choose,source=None):
    if source is not None:
        directory=source/'memory'
        abstract=json.loads((source/'private_skill.json').read_text())['abstract']
        raw=[json.loads(x) for x in (directory/'raw.jsonl').read_text().splitlines()]
    else:
        directory=ROOT/'memory'
        directory.mkdir()
        examples=[]
        for i,r in enumerate(rows):
            key=json.dumps({k:r[k] for k in ('tool','question','returned','before','after','step','effect','future')},ensure_ascii=False)
            feedback='adopt' if r['post'] else ('reject' if r['pre'] else 'neither draft correct')
            examples.append(example(str(i),key,'adopt',feedback))
        abstract=update_memory({'memories':[]},'answer_feedback',examples,directory,gen,capacity=len(rows))[0]['abstract']
        raw=[json.loads(x) for x in (directory/'raw.jsonl').read_text().splitlines()]
    assert len(raw)==len(rows)
    bank=[]
    for record in raw:
        item=json.loads(record['question'])
        assert record['response_is_correct']==(item['effect'] in ('corrected','stayed_correct'))
        bank.append(item)
    counts=Counter((r['tool'],r['effect'],r['future']) for r in bank)
    # Balance edits across changed/unchanged answers and future rescues. Only the fitting partition enters here.
    groups=defaultdict(list)
    for r in bank:
        groups[(r['effect'],r['future'],r['tool'])].append(r)
    selected=[v[i] for i in range(2) for v in groups.values() if len(v)>i]
    contrasts=[{k:(v[:145] if isinstance(v,str) else v) for k,v in r.items()} for r in selected]
    evidence=json.dumps(dict(counts=[[*k,v] for k,v in counts.items()],examples=contrasts,abstract=abstract))
    skills=[]
    for seed,target in [(TRUST,'accepting or discarding a returned observation'),(STOP,'stopping or requesting another tool')]:
        if source is not None:
            from clauses import TRUST_EDITS,STOP_EDITS
            edits=TRUST_EDITS if seed==TRUST else STOP_EDITS
            chosen=choose('Fitting evidence and HCL abstraction: '+evidence,{},
                          'Select the most defensible additional rule for '+target+
                          ' from these complete alternatives. Emphasize observed corrections and harms. '
                          +json.dumps(edits)+' Return one letter.',list(edits))
            instruction=edits[chosen]
        else:
            instruction=gen('Edit a reusable policy for '+target+'. From the paired fit outcomes infer observable conditions '
                            'that distinguish correction, corruption, or later rescue. Give a short operational instruction '
                            'under 70 words. Do not include case names, diagnoses, answers, or a worked example.\nCurrent: '
                            +seed+'\nFit evidence: '+evidence,115).strip()
        if not instruction:
            raise ValueError('empty outcome-guided edit')
        skills.append(append(seed,instruction))
    save('private_skill',dict(trust=skills[0],stop=skills[1],abstract=abstract))
    return bank,skills


def support(bank,tool,q,text):
    pool=[r for r in bank if r['tool']==tool]
    if not pool:
        return 'No fitting observations for this tool.'
    tokens=words(q+' '+text)
    def similarity(row):
        other=words(row['question']+' '+row['returned'])
        return len(tokens & other)/max(1,len(tokens | other))
    neighbors=sorted(pool,key=similarity,reverse=True)[:5]
    return ('Outcomes in similar fit states: '+str(dict(Counter(r['effect'] for r in neighbors)))
            +'. Later tool outcomes: '+str(dict(Counter(r['future'] for r in neighbors)))
            +'. Unobserved means the fit trajectory ended.')
