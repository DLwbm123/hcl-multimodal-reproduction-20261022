"""Build metadata only, reusing immutable images and frozen global roles."""
import collections
import json
import random
from pathlib import Path
from .evaluation import TASKS,qa_kind,validate_gold

SUBROLES=('h_search','retention_anchor','train_replay')
RATIOS={'h_search':.70,'retention_anchor':.15,'train_replay':.15}
VISIBLE_KEYS={'case_id','image','question','output_contract','image_size'}

def read_rows(path):
    with Path(path).open() as f:return [json.loads(line) for line in f if line.strip()]

def write_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def write_rows(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as f:
        for x in rows:f.write(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n')

def inside(root,rel):
    p=Path(rel)
    if p.is_absolute() or '..' in p.parts:raise ValueError('unsafe data path')
    result=(Path(root)/p).resolve()
    if not result.is_relative_to(Path(root).resolve()):raise ValueError('path escaped source root')
    return result

def assign_subroles(records,targets,seed):
    """Groupwise rare-stratum coverage, then deterministic weighted deficits.

    Fractions are preparation choices, not paper defaults or exact row quotas.
    All derived questions from one source group stay together across tasks.
    """
    features=collections.defaultdict(set)
    for x in records:
        if x['role']!='adaptation':continue
        task=x['task'];stratum=targets[x['case_id']]['class_label'] if task=='classification' else x.get('question_type') or 'polyp'
        features[x['group_id']].add((task,stratum))
    pools=collections.defaultdict(set)
    for g,fs in features.items():
        for f in fs:pools[f].add(g)
    ids=sorted(features);random.Random(seed).shuffle(ids);rank={g:n for n,g in enumerate(ids)}
    assigned={};used={s:collections.Counter() for s in SUBROLES}
    totals={f:len(gs) for f,gs in pools.items()}
    def put(g,s):
        assigned[g]=s
        for f in features[g]:used[s][f]+=1
    # Preserve at least one group per stratum/role when possible, without splitting groups.
    for f in sorted(pools,key=lambda f:(totals[f],f)):
        for s in ('retention_anchor','train_replay','h_search'):
            if used[s][f]:continue
            available=sorted(pools[f]-assigned.keys(),key=rank.get)
            if available:put(available[0],s)
    for g in ids:
        if g in assigned:continue
        def deficit(s):return sum((RATIOS[s]*totals[f]-used[s][f])/max(totals[f],1) for f in features[g])
        put(g,max(SUBROLES,key=deficit))
    return assigned

def prepare(source,destination,seed=20261022):
    source=Path(source).resolve();out=Path(destination).resolve()
    if out.exists():raise FileExistsError('refuse overwrite; retain prior frozen preparation')
    frozen=json.loads((source/'audit/SPLIT_FROZEN.json').read_text())
    if frozen['version']!=3:raise ValueError('unexpected source split version')
    manifest=read_rows(source/'audit/unified_manifest.jsonl')
    assets=read_rows(source/'audit/image_assets.jsonl')
    vocabulary=json.loads((source/'audit/class_vocabulary.json').read_text())
    visible={};private={};orig={}
    for task in TASKS:
        for role in ('adaptation','validation','test'):
            for v in read_rows(source/f'inputs/{task}/{role}.jsonl'):
                if set(v)-VISIBLE_KEYS:raise ValueError('unexpected model-visible source fields')
                key=v['case_id']
                if key in visible:raise ValueError('duplicate source case ID')
                visible[key]=v;orig[key]=role
            for t in read_rows(source/f'private/targets/{task}/{role}.jsonl'):
                key=t['case_id']
                if key in private:raise ValueError('duplicate private case ID')
                private[key]=t['target']
    expected={x['case_id'] for x in manifest}
    if len(expected)!=len(manifest) or expected!=set(visible) or expected!=set(private):raise ValueError('source join is not one-to-one')
    global_group={};global_pixel={}
    for x in manifest:
        if orig[x['case_id']]!=x['role']:raise ValueError('frozen role mismatch')
        for key,seen in [(x['group_id'],global_group),(x['pixel_id'],global_pixel)]:
            if key in seen and seen[key]!=x['role']:raise ValueError('frozen source leakage')
            seen[key]=x['role']
    assigned=assign_subroles(manifest,private,seed)
    counts=collections.defaultdict(collections.Counter);groups=collections.defaultdict(set);role_index=[]
    files=collections.defaultdict(list);labels=collections.defaultdict(list);joined_by_role=collections.defaultdict(list)
    paths=set();roles_by_group={};roles_by_pixel={};strata=collections.defaultdict(collections.Counter)
    for x in manifest:
        key=x['case_id'];task=x['task'];s=assigned[x['group_id']] if x['role']=='adaptation' else x['role']
        a=assets[x['asset']];v=visible[key];target=private[key]
        if a['pixel_id']!=x['pixel_id'] or a['group_id']!=x['group_id'] or a['image']!=v['image']:raise ValueError('asset identity mismatch')
        if v['image_size']!={'width':a['width'],'height':a['height']}:raise ValueError('image size metadata mismatch')
        path=inside(source,v['image'])
        if not path.name.startswith('im_'):raise ValueError('image path must be anonymous')
        paths.add(path)
        for identity,seen in [(x['group_id'],roles_by_group),(x['pixel_id'],roles_by_pixel)]:
            if identity in seen and seen[identity]!=s:raise ValueError('derived role leakage')
            seen[identity]=s
        q=v['question']
        if task=='classification':q='Classify this image. Return only one exact label from this fixed vocabulary: '+', '.join(vocabulary)+'.'
        question_type=x.get('question_type')
        meta={'width':a['width'],'height':a['height'],'role':s,'group_id':x['group_id'],'pixel_id':x['pixel_id'],'question_type':question_type}
        if task=='classification':answer=target['class_label'];stratum=answer
        elif task=='polyp_localization':answer={'bbox':target['bbox']};stratum='polyp'
        else:
            answer=target['answer'];meta['answer_kind']=qa_kind(answer,question_type)
            if question_type and question_type.endswith('_set'):meta['answer_kind']='set'
            stratum=question_type or 'text'
        visible_row={'id':key,'question':q,'image':str(path),'visible_context':{'width':a['width'],'height':a['height'],'bbox_coordinate_system':'original_pixel_half_open_xyxy' if task=='polyp_localization' else 'not_applicable'}}
        label_row={'id':key,'answer':answer,'metadata':meta}
        internal={**visible_row,'task_id':key,'task_name':task,'task_type':'medical_'+task,'split':s,**{k:label_row[k] for k in ('answer','metadata')}}
        validate_gold(internal,vocabulary)
        files[(task,s)].append(visible_row);labels[(task,s)].append(label_row);joined_by_role[s].append(internal)
        counts[task][s]+=1;groups[(task,s)].add(x['group_id']);strata[(task,s)][stratum]+=1
        role_index.append({'id':key,'task':task,'role':s,'original_role':x['role'],'group_id':x['group_id'],'pixel_id':x['pixel_id']})
    # Metadata existence check once per unique referenced image; no hashing or image decoding.
    missing=[str(p) for p in paths if not p.is_file()]
    if missing:raise FileNotFoundError(str(missing[:3]))
    for task in TASKS:
        for role in ('validation','test'):
            if counts[task][role]!=frozen['counts'][task][role]:raise ValueError('original locked role changed')
        for role in SUBROLES:
            if not counts[task][role]:raise ValueError('empty derived role')
        if len(groups[(task,'retention_anchor')])<20:raise ValueError('historical anchors under original minimum')
    out.mkdir(parents=True)
    for (task,s),rs in files.items():
        write_rows(out/f'visible/{task}/{s}.jsonl',rs)
        write_rows(out/f'private/{task}/{s}.jsonl',labels[(task,s)])
    write_rows(out/'audit/roles.jsonl',role_index)
    write_json(out/'class_vocabulary.json',vocabulary)
    task_specs=[]
    for task in TASKS:
        task_specs.append({'task_name':task,'task_type':'medical_'+task,'splits':{'train':f'visible/{task}/h_search.jsonl','validation':f'visible/{task}/validation.jsonl','anchor':f'visible/{task}/retention_anchor.jsonl','test':f'visible/{task}/test.jsonl'},'image_root':str(source)})
    write_json(out/'taskstream.json',{'name':'medical_hcl_v1','tasks':task_specs,'roles':'private files are joined only inside StrictMedicalLoader'})
    report={'source_split_version':3,'seed':seed,'original_validation_test_unchanged':True,'adaptation_group_ratios':RATIOS,'counts':{t:dict(c) for t,c in counts.items()},'unique_groups':{t:{s:len(groups[(t,s)]) for s in (*SUBROLES,'validation','test')} for t in TASKS},'strata':{t:{s:dict(strata[(t,s)]) for s in (*SUBROLES,'validation','test')} for t in TASKS},'unique_referenced_images':len(paths),'all_paths_exist':True,'target_join_complete':True,'all_targets_contract_valid':True,'cross_group_roles':0,'cross_pixel_roles':0,'source_images_copied':0,'model_calls':0,'training_runs':0,'patient_isolation':'unverified; no patient IDs','test_labels_used_for_selection':False}
    write_json(out/'audit/preparation_check.json',report)
    write_json(out/'FROZEN.json',{'protocol':'medical_hcl_v1','seed':seed,'counts':report['counts'],'rule':'no role changes from model or test outcomes; immutable source split v3 retained'})
    return report

class StrictMedicalLoader:
    """Join evaluation labels internally; execution visibility is a separate allowlist."""
    def __init__(self,prepared):
        self.root=Path(prepared).resolve()
        self.seed=json.loads((self.root/'FROZEN.json').read_text())['seed']
    def load(self,source):
        if isinstance(source,(str,Path)):
            p=Path(source).resolve()
            if not p.is_relative_to(self.root/'visible'):raise ValueError('loader requires prepared visible manifests')
            rows=read_rows(p);rel=p.relative_to(self.root/'visible');labels=read_rows(self.root/'private'/rel)
            by={x['id']:x for x in labels}
            if len(by)!=len(labels) or {x['id'] for x in rows}!=set(by):raise ValueError('private join mismatch')
            task=rel.parts[0];role=p.stem
            return [{**v,'task_id':v['id'],'task_name':task,'task_type':'medical_'+task,'split':role,'answer':by[v['id']]['answer'],'metadata':by[v['id']]['metadata']} for v in rows]
        return [dict(x) for x in source]
    def load_task_spec(self,spec,*,split,dataset_index=0,limit=None):
        aliases={'train':'h_search','anchor':'retention_anchor','validation':'validation','test':'test'}
        if split not in aliases:raise ValueError('unsupported split; no implicit validation fallback')
        task=spec['task_name'];rows=self.load(self.root/f'visible/{task}/{aliases[split]}.jsonl')
        if limit is not None:
            if type(limit)is not int or limit<1:raise ValueError('limit must be positive')
            # One case per group before repeat QA, retaining deterministic order.
            selected=[];seen=set();buckets=collections.defaultdict(list)
            if aliases[split]=='test':
                rows=sorted(rows,key=lambda x:x['task_id'])
                random.Random(self.seed).shuffle(rows)
            for x in rows:
                stratum='all' if aliases[split]=='test' else x['answer'] if task=='classification' else x['metadata'].get('question_type') or 'polyp'
                buckets[stratum].append(x)
            # Label/type stratification is frozen before model outputs; never score-based.
            for key in sorted(buckets,key=lambda key:(len(buckets[key]),key)):
                for x in buckets[key]:
                    g=x['metadata']['group_id']
                    if g not in seen:
                        selected.append(x);seen.add(g);break
                if len(selected)==limit:break
            if len(selected)<limit:
                for x in rows:
                    g=x['metadata']['group_id']
                    if g not in seen:selected.append(x);seen.add(g)
                    if len(selected)==limit:break
            if len(selected)<limit:raise ValueError('limit exceeds available independent source groups')
            rows=selected
        for x in rows:x['dataset_index']=dataset_index
        return rows

def model_visible(example):
    return {'image':example['image'],'question':example['question'],'visible_context':dict(example['visible_context'])}
