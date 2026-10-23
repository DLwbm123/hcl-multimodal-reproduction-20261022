import os
import json,time,random,shutil
from pathlib import Path
C=Path(os.environ['CAMPAIGN_ROOT']);R=C/'runs/frozen_vqa_confirmation_v1';P=C.parent/'prepared_data';OLD=C/'runs/task_scoped_memory_v1'
assert not R.exists();R.mkdir();[(R/n).mkdir() for n in ['code','tmp','cache']]
now=time.time()
def write(n,x):(R/n).write_text(json.dumps(x,indent=2))
rows=lambda p:[json.loads(x) for x in p.read_text().splitlines()]
cb=json.loads((C/'campaign_budget.json').read_text());assert cb['early_stop_epoch']-now>6600
b=json.loads((OLD/'budget.json').read_text());b.update(started_epoch=now,early_stop_epoch=min(now+6000,cb['early_stop_epoch']),hard_deadline_epoch=min(now+6300,cb['deadline_epoch']),max_calls_per_hcl=1100,max_calls_baselines=850,early_stop_bytes=450000000,max_new_bytes=500000000);write('budget.json',b)
roles={r['id']:r for r in rows(P/'audit/roles.jsonl')};exclude=set()
for f in [C/'runs/semantic_boundary_v1/private_frozen_confirmation_membership.json',C/'runs/learning_confirmation_v1/private_frozen_membership.json',OLD/'private_frozen_membership.json']:
 exclude.update(x['group_id'] for x in json.loads(f.read_text())['cases'])
exclude.update(x['metadata']['group_id'] for x in rows(C/'runs/semantic_boundary_v1/private_screen.jsonl'))
visible=rows(P/'visible/visual_qa/validation.jsonl')
exclude.update(roles[x['id']]['group_id'] for x in sorted(visible,key=lambda x:x['id'])[:4])
learning=json.loads((OLD/'private_learning.json').read_text())
for data in learning.values():
 for es in data.values():exclude.update(x['metadata']['group_id'] for x in es)
shuffled=sorted(visible,key=lambda x:x['id']);random.Random(2026102304).shuffle(shuffled);selected=[];used=set(exclude)
for x in shuffled:
 g=roles[x['id']]['group_id']
 if g in used:continue
 selected.append({'id':x['id'],'group_id':g,'task':'visual_qa'});used.add(g)
 if len(selected)==200:break
assert len(selected)==200
write('private_frozen_membership.json',{'seed':2026102304,'cases':selected,'frozen_before_answer_join':True})
ids={x['id'] for x in selected};answers={x['id']:x for x in rows(P/'private/visual_qa/validation.jsonl') if x['id'] in ids}
joined=[{**v,'task_id':v['id'],'task_name':'visual_qa','task_type':'medical_visual_qa','split':'validation','answer':answers[v['id']]['answer'],'metadata':answers[v['id']]['metadata']} for v in visible if v['id'] in ids]
(R/'private_confirmation.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in joined));shutil.copy2(P/'class_vocabulary.json',R/'class_vocabulary.json')
state=json.loads((OLD/'hcl_scoped/committed_stage_state.json').read_text());assert state['stage']==3
write('private_frozen_state.json',state)
write('protocol.json',{'name':'frozen_vqa_confirmation_v1','tasks':['visual_qa'],'confirmation_per_task':{'visual_qa':200},'arms':['static_no_memory','hcl_scoped'],'frozen_source':'task_scoped_memory_v1 final committed stage3 only; not chosen by confirmation score','purpose':'one independent replication of the frozen task-aware HCL VQA signal; no further learning or policy change','model':'same frozen Qwen2.5-VL-7B BF16, temperature0, decoding/interface unchanged','anchors_per_task':24,'selection_or_learning':False,'test_used':False,'success_gate':'VQA >= Static +5 percentage points AND paired bootstrap95% lower >0; exact McNemar and paired loss/gain always reported; no claim of new cross-task retention evidence','scope':'VQA only; previous 3-task pooled gate failed and remains failed; no new localization groups; no patient independence claim','calls_max':1950,'no_retry_policy':'existing schema failure handling only; other failure fatal; no automatic restart','stop':'single frozen policy, no model/prompt/threshold change after result'})
print('frozen',len(joined),'started',now)
