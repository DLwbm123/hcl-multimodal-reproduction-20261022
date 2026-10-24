"""Bounded local edits and matched whole-bank candidates; no routing changes."""
import copy,re

def local_bank(before,proposed,examples,batch_index):
    result=copy.deepcopy(before)
    old_ids={x['memory_id'] for x in before['raw']}
    order={e['task_id']:i for i,e in enumerate(examples)}
    incoming=[x for x in proposed['raw'] if x['memory_id'] not in old_ids and x.get('internal_source',{}).get('task_id') in order]
    incoming.sort(key=lambda x:order[x['internal_source']['task_id']])
    protected=set()
    tokens=lambda x:set(re.findall(r'[a-z0-9]+',x.lower()))
    for item in incoming[:2]:
        records=result['raw']
        if len(records)<10:records.append(copy.deepcopy(item))
        else:
            query=tokens(item.get('question',''))
            def similarity(i):
                terms=tokens(records[i].get('question',''));u=query|terms
                return len(query&terms)/len(u) if u else 0
            eligible=[i for i,r in enumerate(records) if r['memory_id'] not in protected]
            i=max(eligible,key=lambda i:(similarity(i),-i));records[i]=copy.deepcopy(item)
        protected.add(item['memory_id'])
    role='main_policy' if batch_index==0 else 'failure_avoidance'
    incoming=[x for x in proposed['abstract'] if x['abstract_role']==role]
    assert len(incoming)<=1
    if incoming:
        found=[i for i,x in enumerate(result['abstract']) if x['abstract_role']==role]
        if found:result['abstract'][found[0]]=copy.deepcopy(incoming[0])
        else:result['abstract'].append(copy.deepcopy(incoming[0]))
    assert len(result['raw'])<=10 and len(result['abstract'])<=2
    assert result['anchor']==before['anchor'] and result['skills']==before['skills']
    return result

def candidates(before,proposed,examples,batch_index,local):
    edited=local_bank(before,proposed,examples,batch_index) if local else proposed
    alternatives=[]
    for name,keys in [('raw_only',['raw']),('abstract_only',['abstract']),('raw_and_abstract',['raw','abstract'])]:
        state=copy.deepcopy(before)
        for key in keys:state[key]=copy.deepcopy(edited[key])
        alternatives.append((name,state))
    return alternatives

def safe_metrics(result):
    return {part:{k:v for k,v in result[part].items() if k in ('correct','total','primary_score','format_compliance_rate','forget_count','recovered_count','reference_correct')} for part in ('current_task_metrics','historical_anchor_metrics')}

def train_local_batch(pipe,examples,selection,anchors,arm,stage,batch_index,record,guard):
    before=pipe.memory.snapshot();gate=selection+anchors
    assert not {e['task_id'] for e in examples}&{e['task_id'] for e in gate}
    assert all(e['metadata']['stream_stage']!=stage for e in anchors)
    context=pipe._historical_anchor_context(anchors)
    assert not context['required'] or context['ready']
    def evaluate(predictions,reference=None):
        return pipe._evaluate_gate(gate,predictions,current_val_examples=selection,historical_anchor_examples=anchors,split='selection:'+stage,reference_predictions=reference)
    guard();reference_predictions=pipe._predict(gate,update_memory=False,progress_label='local:reference')
    reference=evaluate(reference_predictions);assert pipe.memory.snapshot()==before
    try:
        pipe._evaluate_train_batch(examples,split=f'train:{stage}:{batch_index}')
        proposed=pipe.memory.snapshot()
        choices=candidates(before,proposed,examples,batch_index,local=arm!='bulk')
        pipe.memory.restore(before)
        rows=[];eligible=[]
        for i,(name,state) in enumerate(choices):
            guard();pipe.memory.restore(state)
            predictions=pipe._predict(gate,update_memory=False,progress_label='local:'+name)
            assert pipe.memory.snapshot()==state
            result=evaluate(predictions,reference_predictions)
            accepted,reasons=pipe._memory_candidate_decision(reference,result,anchor_context=context)
            row={'index':i,'scope':name,'eligible':accepted,'reasons':reasons,'metrics':safe_metrics(result),'raw_count':len(state['raw']),'abstract_count':len(state['abstract'])}
            rows.append(row)
            if accepted:eligible.append((int(result['current_task_metrics']['correct']),-int(result['historical_anchor_metrics'].get('forget_count',0)),-i,i))
        winner=max(eligible)[-1] if eligible else None
        pipe.memory.restore(choices[winner][1] if winner is not None else before)
        record({'stage':stage,'batch':batch_index+1,'arm':arm,'accepted':winner is not None,'chosen_index':winner,'reference':safe_metrics(reference),'candidates':rows,'memory':pipe.memory.summary()})
    except BaseException:
        pipe.memory.restore(before)
        raise
