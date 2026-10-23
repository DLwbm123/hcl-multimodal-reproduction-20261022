"""Token-budgeted proposer input. Only current learning records enter here."""
import json

def bounded(value,chars=400,items=8):
    if isinstance(value,str):return value if len(value)<=chars else value[:chars]+' [truncated]'
    if isinstance(value,list):return [bounded(x,chars,items) for x in value[:items]]+([{'omitted_items':len(value)-items}] if len(value)>items else [])
    if isinstance(value,dict):return {k:bounded(v,chars,items) for k,v in value.items()}
    return value

def feedback_rows(train,training,gold):
    out=[]
    for c,r in zip(train,training):
        answers=gold[c['id']]['calculator_answers']
        out.append({'prompt':c['prompt'],'trajectory':[bounded(e) for e in r['trace'] if e.get('name')!='read_facts'][-12:],
                    'submitted_answer':bounded(r['answer']), 'utility':r['utility'],
                    'correct_feedback':[{'name':a['name'],'final_answer':a['final_answer']} for a in answers]})
    return out

def pack_request(context,index,previous,system,count_tokens,max_input=20000):
    from langchain_core.messages import SystemMessage,HumanMessage
    context=json.loads(json.dumps(context));cuts=0
    def request():return [SystemMessage(content=system),HumanMessage(content=json.dumps({'proposal_index':index,'previous_proposals':previous,**context}))]
    while True:
        req=request();n=count_tokens(req)
        if n<=max_input:return req,{'input_tokens':n,'cap':max_input,'removed_trajectory_events':cuts,'training_cases':len(context['training'])}
        eligible=[t for t in context['training'] if t['trajectory']]
        if not eligible:raise ValueError('essential learning inputs exceed proposer token budget; no silent truncation')
        max(eligible,key=lambda t:len(json.dumps(t['trajectory'])))['trajectory'].pop(0);cuts+=1
