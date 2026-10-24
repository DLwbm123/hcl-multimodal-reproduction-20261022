"""Offline full-denominator audit; exports aggregate counts only, no model calls."""
import json,math,sys
from pathlib import Path
R=Path(sys.argv[1])
def load(p):return json.loads((R/p).read_text())
def lines(p):return [json.loads(x) for x in (R/p).read_text().splitlines()]
def values(role,block):return {x['task_id']:bool(x['success']) for x in load(f'{role}/private_eval_{block}.json')['case_results']}
def paired(before,after,ids=None):
 ids=set(before) if ids is None else set(ids);assert ids<=set(before) and ids<=set(after)
 both=sum(before[k] and after[k] for k in ids);gain=sum(not before[k] and after[k] for k in ids);loss=sum(before[k] and not after[k] for k in ids);neither=len(ids)-both-gain-loss;n=gain+loss
 p=min(1.,2*sum(math.comb(n,i) for i in range(min(gain,loss)+1))/2**n) if n else 1.
 return {'n':len(ids),'before_correct':both+loss,'after_correct':both+gain,'both_correct':both,'gained':gain,'lost':loss,'both_wrong':neither,'exact_mcnemar_two_sided_p':p}

def main():
 p1,p2=load('p1/RESULT.json'),load('p2/RESULT.json'); data=load('private_data.json')
 exits={role:load(role+'/EXIT.json') for role in ['p1','p2']};assert all(v['exit_code']==0 for v in exits.values())
 out={'model_calls_during_analysis':0,'full_denominators':True,'scope':'development diagnosis; unadjusted exact paired p values are descriptive','execution':{'roles':exits,'supervisor':load('SUPERVISOR_EXIT.json')}}
 p1out={'chosen':p1['chosen'],'selection':{},'review':paired(values('p1','after_B_review'),values('p1','chosen_review')),'promotion':p1['promotion'],'B_true_to_shifted':paired(values('p1','B_true_image'),values('p1','B_shifted_image')),'missing_flags':{k:{split:sum(str(e['question']).strip().lower() in {'','nan','null','none'} or str(e['answer']).strip().lower() in {'','nan','null'} for e in data[k][split]) for split in ['train','selection','review']} for k in ['A','B','C']}}
 base=values('p1','after_B_selection')
 for name,s in p1['selection'].items():
  block='after_B_selection' if name=='after_B' else name+'_selection'
  p1out['selection'][name]={'quality':s['quality'],'format_compliance':s['metrics']['format_compliance_rate'],'paired':paired(base,values('p1',block)),'retrieval':s['retrieval']}
 out['p1']=p1out
 A={e['task_id'] for e in data['A']['review']};B={e['task_id'] for e in data['B']['review']}
 common=values('p2','common_review');empty=values('p2','empty_A_review');gains={k for k in A if common[k] and not empty[k]}
 assert len(A)==len(B)==24 and len(gains)==p2['new_A_review_count']
 p2out={'empty_to_common_A':paired(empty,common,A),'common_A_correct':sum(common[k] for k in A),'common_B_correct':sum(common[k] for k in B),'new_A_reference_count':len(gains),'blocks':[],'replicated_mechanism':p2['replicated_mechanism']}
 for block in p2['blocks']:
  b=block['block']; row={'block':b,'mechanism_pattern_passed':block['mechanism_pattern_passed'],'policies':{}}
  for name,policy in block['policies'].items():
   if not policy['accepted']:current=common
   else:
    same_zero=name.endswith('inf') and policy['chosen_index']==block['policies'][name.replace('inf','0')]['chosen_index']
    source=name.replace('inf','0') if same_zero else name
    current=values('p2',f'block{b}_{source}_review')
   row['policies'][name]={'accepted':policy['accepted'],'chosen_index':policy['chosen_index'],'A':paired(common,current,A),'B':paired(common,current,B),'joint':paired(common,current,A|B),'new_A_lost':sum(not current[k] for k in gains),'memory_change':policy['memory_change'],'candidate_decisions':policy['candidate_decisions'],'candidate_metrics':policy['candidate_metrics'],'retrieval':policy['review_summary']['retrieval']}
   assert row['policies'][name]['new_A_lost']==policy['new_A_lost']
  p2out['blocks'].append(row)
 out['p2']=p2out
 audit={}
 for role in ['p1','p2']:
  costs=lines(role+'/call_costs.jsonl'); readonly=lines(role+'/readonly_checks.jsonl');train=lines(role+'/private_train_interactions.jsonl')
  assert len(costs)==exits[role]['calls'] and all(x['success'] for x in costs) and all(x['passed'] for x in readonly)
  final=[c for c in costs if c['phase']=='final_answer'];assert all(c['image_supplied'] for c in final)
  byblock={}
  for x in train:byblock[x['block']]=byblock.get(x['block'],0)+1
  assert all(n==24 for n in byblock.values())
  audit[role]={'call_records':len(costs),'all_calls_succeeded':True,'final_answer_calls':len(final),'all_final_calls_had_image':True,'readonly_evaluations':len(readonly),'train_interactions_by_block':byblock,'sum_call_seconds':sum(x['seconds'] for x in costs)}
 out['audit']=audit
 (R/'AGGREGATE_REPORT.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({'calls':sum(v['calls'] for v in exits.values()),'p1_review':p1out['review'],'P2':[{'block':b['block'],'policies':{n:{'A':v['A']['after_correct'],'B':v['B']['after_correct'],'new_A_lost':v['new_A_lost'],'accepted':v['accepted']} for n,v in b['policies'].items()}} for b in p2out['blocks']],'audit':audit}))
if __name__=='__main__':main()
