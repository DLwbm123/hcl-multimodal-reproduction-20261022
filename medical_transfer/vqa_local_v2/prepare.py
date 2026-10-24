"""Reuse declared development data; reserve never-used original replay groups as holdout."""
import json,os,time,sys,collections,copy
from pathlib import Path
R=Path(os.environ['RUN_ROOT']);P=Path(os.environ['PREPARED_DATA']);V=R.parent/'vqa_continual_v1'
sys.path.insert(0,str(R/'code'))
from selection import allocate
plan=json.loads((R/'protocol.json').read_text())
assert not (R/'private_membership.json').exists()
rows=lambda p:[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
roles={x['id']:x for x in rows(P/'audit/roles.jsonl')}
previous=json.loads((V/'private_membership.json').read_text());excluded=set();pixels=set()
def collect(obj):
    if isinstance(obj,dict):
        if obj.get('group_id'):excluded.add(obj['group_id'])
        if obj.get('pixel_id'):pixels.add(obj['pixel_id'])
        for k in ('id','task_id'):
            if obj.get(k) in roles:
                r=roles[obj[k]];excluded.add(r['group_id'])
                if r.get('pixel_id'):pixels.add(r['pixel_id'])
        for v in obj.values():collect(v)
    elif isinstance(obj,list):
        for v in obj:collect(v)
for f in previous['exclusion_sources']:
    p=Path(f);collect(rows(p) if p.suffix=='.jsonl' else json.loads(p.read_text()))
for task in ['visual_qa','classification','polyp_localization']:
    for x in sorted(rows(P/f'visible/{task}/validation.jsonl'),key=lambda x:x['id'])[:4]:collect(x)
collect(previous)
# Read metadata only for allocation; associate answer fields after durable membership.
meta={x['id']:x['metadata'] for x in rows(P/'private/visual_qa/train_replay.jsonl')}
visible=rows(P/'visible/visual_qa/train_replay.jsonl');pools={};quotas={};availability={}
for stage,types in plan['stages'].items():
    for q in types:
        pool=collections.defaultdict(list)
        for v in visible:
            m=meta[v['id']]
            if m['question_type']==q and m['group_id'] not in excluded and m['pixel_id'] not in pixels:
                pool[m['group_id']].append({'id':v['id'],'group_id':m['group_id'],'pixel_id':m['pixel_id'],'stage':stage,'split':'report','source_role':'train_replay','question_type':q})
        pools[stage,q]=pool;quotas[stage,q]=36;availability[stage+'/'+q]=len(pool)
allocated=allocate(pools,quotas,plan['seed']);fresh=[]
for stage in plan['order']:
    a,b=[allocated[stage,q] for q in plan['stages'][stage]]
    fresh.extend(x for pair in zip(a,b) for x in pair)
reused=[copy.deepcopy(x) for x in previous['cases'] if x['split']!='report']
membership=reused+fresh
assert len(membership)==504 and len({x['group_id'] for x in membership})==504 and len({x['pixel_id'] for x in membership})==504
assert not {x['group_id'] for x in fresh}&excluded and not {x['pixel_id'] for x in fresh}&pixels
(R/'private_membership.json').write_text(json.dumps({'cases':membership,'frozen_at':time.time(),'seed':plan['seed'],'excluded_report_groups':len(excluded),'original_role_reassignment':'216 train_replay groups reserved report-only; never training in any run','answers_used_for_selection':False},indent=2))
data=json.loads((V/'private_data.json').read_text())
labels={x['id']:x for x in rows(P/'private/visual_qa/train_replay.jsonl')};vmap={x['id']:x for x in visible}
for stage in plan['order']:data[stage]['report']=[]
for m in fresh:
    v=vmap[m['id']];lab=labels[m['id']]
    data[m['stage']]['report'].append({**v,'task_id':m['id'],'task_name':'visual_qa','task_type':'medical_visual_qa','split':'reserved_report','answer':lab['answer'],'metadata':{**lab['metadata'],'original_role':'train_replay','role':'reserved_report','stream_stage':m['stage']}})
(R/'private_data.json').write_text(json.dumps(data))
receipt={'passed':True,'groups':504,'reused_development_groups':288,'fresh_report_groups':216,'report_excluded_previous_groups':len(excluded),'report_availability':availability,'group_and_pixel_disjoint':True,'test_opened':False,'targets_associated_after_membership':True,'original_role_reassignment':'unused train_replay to reserved_report','historical_source_code_train_replay_usage':'data preparation/audit only; no learning entrypoint uses this role','model_calls':0}
(R/'DATA_RECEIPT.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
