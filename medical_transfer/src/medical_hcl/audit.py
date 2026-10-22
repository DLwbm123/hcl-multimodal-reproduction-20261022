"""Independent reader of emitted metadata; no image/model calls or split changes."""
import collections,json
from pathlib import Path
from .data import read_rows,StrictMedicalLoader,model_visible
from .evaluation import TASKS,validate_gold

def audit(prepared,source):
    root=Path(prepared);source=Path(source)
    original={x['case_id']:x for x in read_rows(source/'audit/unified_manifest.jsonl')}
    roles=read_rows(root/'audit/roles.jsonl');vocab=json.loads((root/'class_vocabulary.json').read_text())
    if len(roles)!=len(original) or len({x['id'] for x in roles})!=len(roles):raise ValueError('case coverage changed')
    seen_group={};seen_pixel={};byid={};expected=collections.Counter()
    for x in roles:
        old=original[x['id']]
        if (x['group_id'],x['pixel_id'],x['task'],x['original_role'])!=(old['group_id'],old['pixel_id'],old['task'],old['role']):raise ValueError('identity drift')
        if old['role'] in ('validation','test') and x['role']!=old['role']:raise ValueError('heldout role changed')
        if old['role']=='adaptation' and x['role'] not in ('h_search','retention_anchor','train_replay'):raise ValueError('invalid derived role')
        for identity,seen in ((x['group_id'],seen_group),(x['pixel_id'],seen_pixel)):
            if identity in seen and seen[identity]!=x['role']:raise ValueError('source crosses roles')
            seen[identity]=x['role']
        expected[x['task'],x['role']]+=1;byid[x['id']]=x
    loader=StrictMedicalLoader(root);actual=collections.Counter();unique=set()
    for task in TASKS:
        for role in ('h_search','retention_anchor','train_replay','validation','test'):
            p=root/f'visible/{task}/{role}.jsonl'
            for row in read_rows(p):
                if set(row)!={'id','image','question','visible_context'}:raise ValueError('unexpected visible key')
                if not Path(row['image']).name.startswith('im_'):raise ValueError('class-bearing image path')
            rows=loader.load(p)
            for e in rows:
                k=e['task_id'];m=e['metadata'];x=byid[k]
                if k in unique:raise ValueError('case emitted twice')
                unique.add(k)
                if (task,role,m['group_id'],m['pixel_id'])!=(x['task'],x['role'],x['group_id'],x['pixel_id']):raise ValueError('emitted role mismatch')
                validate_gold(e,vocab)
                if set(model_visible(e))!={'image','question','visible_context'}:raise ValueError('visibility violation')
                actual[task,role]+=1
    if actual!=expected or unique!=set(original):raise ValueError('emitted denominator changed')
    return {'passed':True,'cases':len(unique),'groups':len(seen_group),'pixels':len(seen_pixel),'cross_role_groups':0,'cross_role_pixels':0,'source_identity_preserved':True,'validation_test_preserved':True,'model_visible_field_allowlist_passed':True,'private_targets_complete':True,'target_contracts_passed':True,'image_content_rescanned':False,'patient_level_isolation_verified':False,'model_calls':0}
