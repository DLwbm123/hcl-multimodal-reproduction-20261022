"""Audit actual task-routed raw memory, paired acquisition and small A retention."""
from pathlib import Path
from collections import Counter
import json,os,sys
from task_memory import ARMS,FINAL,select,context,gate
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'medrax_hcl_v5'))
from logic import paired

def main():
 r=Path(os.environ['R981_ROOT']);read=lambda n:json.loads((r/n).read_text());f=read('FINISHED.json');rows=read('private/episodes.json');items=read('private/inputs.json');states=read('private/states.json');support=read('private/support_selection.json');gold=read('private/sealed_development_gold.json');jobs=read('private/requests_metadata.json');events=[json.loads(x) for x in (r/'private/events.jsonl').read_text().splitlines()]
 assert f['complete'] and f['fatal'] is None and len(rows)==48 and all(x['status']=='completed' and x['predicted'] in list('ABCDEF') for x in rows)
 assert len(states['H_A']['raw'])==24 and len(states['H_B']['raw'])==48 and states['H_B']['raw'][:24]==states['H_A']['raw'] and states['H0_static']['raw']==[]
 train={x['internal_source']['task_id'] for x in states['H_B']['raw']};target={x['id'] for x in items};assert len(train)==48 and len(target)==9 and not train&target
 train_images={y for x in states['H_B']['raw'] for y in x['images']};target_images={y for x in items for y in x['images']};assert not train_images&target_images
 v9=r.parent/'medrax_learning_qualification_v9';reserved={x['full_question_id'] for x in json.loads((v9/'private/members.json').read_text()) if x['role']=='confirmation_reserved'};assert len(reserved)==96 and not reserved&(train|target)
 v10=r.parent/'medrax_evidence_memory_v10';old=json.loads((v10/'private/results.json').read_text());old={x['id']:x for x in old['H0_static_r1']};orig=json.loads((v10/'private/inputs.json').read_text());assert [x['id'] for x in items if x['stage']=='A']==[x['id'] for x in orig if old[x['id']]['correct']][:3]
 assert [x['id'] for x in items if x['stage']=='B']==[x['id'] for x in json.loads((r.parent/'medrax_visual_focus_v15/private/inputs.json').read_text())]
 expected=[]
 for b in [1,2]:
  for stage,arms in [('B',ARMS),('A',['H_A','H_B'])]:
   panel=[x for x in items if x['stage']==stage]
   for i,x in enumerate(panel if b==1 else reversed(panel)):
    offset=(i+b-1)%len(arms);expected.extend((x['id'],stage+'_'+a+'_r'+str(b)) for a in arms[offset:]+arms[:offset])
 assert [(x['id'],x['state']) for x in rows]==expected
 starts=[e for e in events if e['kind']=='controller_started'];ends=[e for e in events if e['kind']=='controller_finished'];tools=[e for e in events if e['kind']=='tool_finished'];per={}
 for e in starts:per.setdefault(e['episode'],[]).append(e)
 assert len(per)==48 and min(e['time'] for e in starts)>read('STATES_FROZEN.json')['time']
 lookup={x['id']:x for x in items};firsts={};selected_sources={'A':set(),'B':set()};contexts=Counter()
 for i,row in enumerate(rows):
  x=lookup[row['id']];arm=row['state'].split('_',1)[1].rsplit('_r',1)[0];selected=select(states[arm],x['question'],x['stage'],arm);text=context(selected);assert row['retrieved_ids']==support[x['id']][arm]==[v['memory_id'] for v in selected] and row['context_text']==text and row['context_chars']==len(text)
  assert len(selected)==(0 if arm=='H0_static' else 2)
  for v in selected:selected_sources[v['internal_source']['task_name'].rsplit('_',1)[1]].add(v['memory_id'])
  contexts[x['stage']+'_'+arm]+=bool(text);actual=[jobs[e['key']] for e in per[i]];first=actual[0];figure=''.join(Path(p).name+' located at '+p+'\n' for p in x['images']);native='Answer this question correctly using chain of thought reasoning and carefully evaluating choices. Solve using our own vision and reasoning and thenuse tools to complement your reasoning. Trust your own judgement over any tools.\n'+x['question']+'\n'+figure
  assert first['messages'][0]['content'][0]=={'type':'text','text':native+text} and first['image_count']==len(x['images']) and [p['path'] for p in first['messages'][0]['content'] if p['type']=='image']==x['images']
  humans=[m['content'] for m in actual[-1]['messages'] if m['role']=='user'];assert len(humans)==2 and humans[-1]==FINAL;firsts[(row['id'],row['state'])]=first
 ids={s:[x['id'] for x in items if x['stage']==s] for s in ['A','B']};by={label:{x['id']:x for x in rows if x['state']==label} for label in f['metrics']};pairs=[];same_A=0;different_B=0
 for b in [1,2]:
  for k in ids['A']:assert firsts[(k,'A_H_A_r'+str(b))]==firsts[(k,'A_H_B_r'+str(b))];same_A+=1
  for k in ids['B']:assert by['B_H_A_r'+str(b)][k]['context_text']!=by['B_H_B_r'+str(b)][k]['context_text'];different_B+=1
  get=lambda s,a:list(by[s+'_'+a+'_r'+str(b)].values());pairs.append({'B_H0_to_HB':paired(get('B','H0_static'),get('B','H_B'),ids['B'],gold),'B_HA_to_HB':paired(get('B','H_A'),get('B','H_B'),ids['B'],gold),'A_HA_to_HB':paired(get('A','H_A'),get('A','H_B'),ids['A'],gold)})
 for label,v in f['metrics'].items():assert sum(x['predicted']==gold[k] for k,x in by[label].items())==v['correct']
 assert pairs==f['paired_per_repeat'];g=gate(pairs,by,ids,gold,True);assert {k:v for k,v in g.items() if k!='repeated_gain_ids'}==f['acquisition_and_small_retention_gate']
 assert len(starts)==len(ends)==read('private/call_count.json')==144 and all(x['status']=='ok' for x in ends);responses=[json.loads(p.read_text()) for p in (r/'private').glob('response_*.json')];assert len(responses)==144;usage=Counter()
 for e in ends:
  for u in e.get('usage',[]):usage.update(u)
 v={'audit_passed':True,'physical_episodes':48,'training_source_cases':48,'confirmation_excluded':96,'source_target_ID_and_known_images_disjoint':True,'selection_and_48_episode_order_reconstructed':True,'actual_initial_memory_payload_and_current_only_images_verified':48,'original_two_turn_final_scoring_verified':48,'B_HA_HB_case_repeat_pairs_with_actual_different_context':different_B,'A_HA_HB_case_repeat_pairs_with_identical_full_initial_payload':same_A,'memory_exposure_episodes':dict(contexts),'selected_distinct_source_counts':{k:len(x) for k,x in selected_sources.items()},'all_paired_tables_and_gate_recomputed':True,'overall_gate_passed':g['pass'],'B_acquisition_passed':g['acquisition_pass'],'small_A_retention_passed':g['small_A_retention_pass'],'repeated_B_gains':g['repeated_B_gains_with_actual_context_update'],'repeat_final_option_changes':{s+'_'+a:sum(by[s+'_'+a+'_r1'][k]['predicted']!=by[s+'_'+a+'_r2'][k]['predicted'] for k in ids[s]) for s,arms in [('B',ARMS),('A',['H_A','H_B'])] for a in arms},'B_final_option_disagreement_across_arms_case_repeat_pairs':sum(len({by['B_'+a+'_r'+str(b)][k]['predicted'] for a in ARMS})>1 for b in [1,2] for k in ids['B']),'controller_calls':144,'memory_generation_calls':0,'usage':dict(usage),'completed_request_seconds':sum(x.get('seconds',0) for x in responses),'visible_429_response_count':sum('429 Too Many Requests' in x.get('stderr','') for x in responses),'tool_counts':dict(Counter(x['name'] for x in tools)),'tool_status_counts':dict(Counter(x['status'] for x in tools)),'original_wall_seconds':f['seconds_since_original_start'],'tool_load_seconds_from_worker_start':read('TOOLS_LOADED.json')['time']-read('STARTED.json')['time'],'target_execution_seconds':f['time']-read('TOOLS_LOADED.json')['time'],'final_guard':read('RUN_GUARD.json'),'skill_evolution_enabled':False,'parameter_or_clinical_retention_established':False};(r/'AUDIT.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))

if __name__=='__main__':main()
