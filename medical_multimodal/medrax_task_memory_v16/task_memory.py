"""Task-routed raw HCL memory: actual B-state update, no generated program."""
import json,re
ARMS=['H0_static','H_A','H_B']
FINAL='If you had to choose the best option, only respond with the letter of choice (only one of A, B, C, D, E, F)'
STOP=set('the and for with from this that these those which what based provided following most likely correct answer option image images imaging chest xray radiograph radiographs findings patient figure'.split())

def stem(text):
    m=re.search(r'(?m)^\s*[A-F][.)]\s+',text);return (text[:m.start()] if m else text).strip()

def option_text(question,label):
    ms=list(re.finditer(r'(?m)^\s*([A-F])[.)]\s+',question));out={m.group(1):question[m.end():ms[i+1].start() if i+1<len(ms) else len(question)].strip() for i,m in enumerate(ms)};assert label in out and out[label];return out[label]

def select(snapshot,question,stage,arm):
    if arm=='H0_static':return []
    task='A' if stage=='A' or arm=='H_A' else 'B';bank=[r for r in snapshot['raw'] if r['internal_source']['task_name']=='medical_agent_'+task];assert len(bank)==24
    terms=lambda s:{x for x in re.findall('[a-z0-9]+',stem(s).lower()) if len(x)>2 and x not in STOP};q=terms(question)
    def score(r):
        t=terms(r['question']);u=q|t;return (-(len(q&t)/len(u) if u else 0),r['memory_id'])
    return sorted(bank,key=score)[:2]

def context(records):
    if not records:return ''
    payload=[{'historical_question_stem':stem(r['question'])[:1500],'historical_attempt_option_text':option_text(r['question'],r['response'])[:1500],'unadjudicated_benchmark_target_text':option_text(r['question'],r['feedback'])[:1500]} for r in records]
    return '\nTwo historical training examples from DIFFERENT cases. Their benchmark feedback was NOT independently clinically adjudicated and can be ambiguous or wrong. Use them as conditional past experience, never as facts or diagnoses about the current patient. Do not copy an answer, invent absent findings, or infer chronology or causation from historical supervision. Current images and the original question remain primary.\n'+json.dumps(payload,ensure_ascii=False)

def gate(pairs,by,ids,gold,complete):
    acquired=[k for k in ids['B'] if all(by['B_H0_static_r'+str(b)][k].get('predicted')!=gold[k] and by['B_H_B_r'+str(b)][k].get('predicted')==gold[k] and len(by['B_H_B_r'+str(b)][k].get('retrieved_ids',[]))==2 and by['B_H_B_r'+str(b)][k].get('context_text')!=by['B_H_A_r'+str(b)][k].get('context_text') for b in [1,2])]
    acquisition=complete and len(acquired)>=2 and all(x['B_H0_to_HB']['gain']-x['B_H0_to_HB']['loss']>=2 and x['B_H0_to_HB']['loss']<=1 and x['B_HA_to_HB']['gain']-x['B_HA_to_HB']['loss']>=1 for x in pairs)
    retention=complete and all(x['A_HA_to_HB']['loss']==0 and x['A_HA_to_HB']['gain']-x['A_HA_to_HB']['loss']>=0 and sum(r.get('predicted')==gold[k] for k,r in by['A_H_A_r'+str(b)].items())>=2 and sum(r.get('predicted')==gold[k] for k,r in by['A_H_B_r'+str(b)].items())>=2 for b,x in enumerate(pairs,1))
    return {'pass':bool(acquisition and retention),'acquisition_pass':bool(acquisition),'small_A_retention_pass':bool(retention),'repeated_B_gains_with_actual_context_update':len(acquired),'repeated_gain_ids':acquired,'development_only':True,'A_context_is_identical_by_task_routing_not_parameter_retention':True}
