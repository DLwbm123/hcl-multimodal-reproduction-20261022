"""Frozen source selection and semantic rendering for visual episodic memory."""
import json,re
ARMS=['H0_static','history_text','history_image','history_mismatch']
STOP=set('the and for with from this that these those which what based provided following most likely correct answer option image images imaging chest xray radiograph radiographs findings patient figure'.split())

def terms(text):return {x for x in re.findall('[a-z0-9]+',text.lower()) if len(x)>2 and x not in STOP}

def choose(records,question):
    q=terms(question)
    def rank(r):
        t=terms(r['question']);u=q|t
        return (-(len(q&t)/len(u) if u else 0),r['memory_id'])
    selected=sorted(records,key=rank)[0]
    i=records.index(selected)
    return selected,records[(i+1)%len(records)]

def option_text(question,label):
    matches=list(re.finditer(r'(?m)^\s*([A-F])[.)]\s+',question));out={}
    for i,m in enumerate(matches):
        assert m.group(1) not in out
        out[m.group(1)]=question[m.end():matches[i+1].start() if i+1<len(matches) else len(question)].strip()
    assert label in out and out[label]
    return out[label]

def context(record,has_history_image):
    payload={'historical_question':record['question'],'historical_attempt_option_text':option_text(record['question'],record['response']),'unadjudicated_benchmark_target_text':option_text(record['question'],record['feedback'])}
    text='\nHistorical training example from a different case. Its benchmark target is unadjudicated supervision and can be ambiguous or wrong. Use it only as conditional experience; do not copy its diagnosis or invent findings for the current case. Current-case images and evidence take priority.\n'+json.dumps(payload,ensure_ascii=False)
    if has_history_image:text+='\nThe additional image after the current-case image is presented as the historical example. It is NOT a current patient image. Do not send it to tools; tools may inspect only current-case images.\n'
    return text

def gate(pairs,by,ids,gold,complete):
    acquired=[k for k in ids if all(by['H0_static_r'+str(i)][k].get('predicted')!=gold[k] and by['history_image_r'+str(i)][k].get('predicted')==gold[k] for i in [1,2])]
    ok=complete and len(acquired)>=2 and all(p['H0_to_image']['gain']-p['H0_to_image']['loss']>=2 and p['H0_to_image']['loss']<=1 and p['text_to_image']['gain']-p['text_to_image']['loss']>=1 and p['mismatch_to_image']['gain']-p['mismatch_to_image']['loss']>=1 for p in pairs)
    return {'pass':bool(ok),'repeat_observed_acquired_count':len(acquired),'repeat_observed_acquired_ids':acquired,'development_only':True}
