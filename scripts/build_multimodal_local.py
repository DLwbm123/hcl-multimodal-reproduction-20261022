"""COCO2014/RefCOCO-UNC/VQAv2 deterministic, image-disjoint local subsets.
Raw archives are immutable; samples are not claimed to be paper IDs.
"""
import collections
import io
import json
import os
import pickle
import random
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(os.environ['REPRO_ROOT'])
SEED = 20261022
COUNTS = {'train': 20, 'validation': 10, 'test': 50}

class BasicUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError(f'Unexpected pickle global: {module}.{name}')

def archive_json(path, member):
    with zipfile.ZipFile(path) as z:
        return json.loads(z.read(member))

def build(*, counts=None, profile="local_image_disjoint", native_pixels=False, excluded_test_images=()):
    counts = counts or COUNTS
    excluded_test_images = set(excluded_test_images)
    raw = ROOT / 'raw'
    coco = archive_json(raw/'annotations_trainval2014.zip', 'annotations/instances_val2014.json')
    caps = archive_json(raw/'annotations_trainval2014.zip', 'annotations/captions_val2014.json')
    with zipfile.ZipFile(raw/'refcoco.zip') as z:
        refs = BasicUnpickler(io.BytesIO(z.read('refcoco/refs(unc).p')), encoding='latin1').load()
        refinst = json.loads(z.read('refcoco/instances.json'))
    questions = archive_json(raw/'vqa_questions.zip', 'v2_OpenEnded_mscoco_val2014_questions.json')['questions']
    answers = archive_json(raw/'vqa_annotations.zip', 'v2_mscoco_val2014_annotations.json')['annotations']
    images = {x['id']: x for x in coco['images']}
    refimages = {x['id']: x for x in refinst['images']}
    categories = {x['id']: x['name'] for x in coco['categories']}
    anns = collections.defaultdict(list)
    for a in coco['annotations']:
        if not a.get('iscrowd'): anns[a['image_id']].append(a)
    capby = collections.defaultdict(list)
    for c in caps['annotations']: capby[c['image_id']].append(c['caption'])
    refanns = {x['id']: x for x in refinst['annotations']}
    qab = {x['question_id']: x for x in answers}
    # Assign each image to one split globally across all four tasks.
    ids = sorted(set(images) | set(refimages))
    random.Random(SEED).shuffle(ids)
    splitby = {i: ('train' if n % 10 < 6 else 'validation' if n % 10 < 8 else 'test') for n,i in enumerate(ids)}
    pools = {t: [] for t in ['coco_detection','coco_caption','refcoco_grounding','vqav2']}
    def base(i, identity, info):
        filename = info.get('file_name', '')
        # RefCOCO uses train2014; instance metadata may have no file_name.
        if not filename: filename = f'COCO_train2014_{i:012d}.jpg'
        return dict(id=str(identity), image=str(ROOT/'images'/filename), metadata={'image_id':i}, visible_context={'width':info['width'],'height':info['height'],'bbox_coordinate_system':'qwen_0_1000'})
    for i in sorted(images):
        info = images[i]
        if anns[i]:
            category = sorted({a['category_id'] for a in anns[i]})[0]
            label = categories[category]
            row = base(i,f'det-{i}-{category}',info)
            row['question'] = f'Locate every {label} in the image. Return only a JSON array of objects with label and bbox [x1,y1,x2,y2], coordinates on a 0-1000 scale.'
            row['answer'] = [{'label':label,'bbox':[a['bbox'][0],a['bbox'][1],a['bbox'][0]+a['bbox'][2],a['bbox'][1]+a['bbox'][3]]} for a in anns[i] if a['category_id']==category]
            row['metadata'].update(answer_bbox_coordinate_system='pixel_xyxy',prediction_bbox_coordinate_system='qwen_0_1000', queried_category=label)
            pools['coco_detection'].append(row)
        if capby[i]:
            row=base(i,f'caption-{i}',info)
            row.update(question='Describe this image in one concise sentence. Output only the caption.',answer=capby[i][0])
            row['metadata']['answers']=capby[i]
            pools['coco_caption'].append(row)
    for r in sorted(refs,key=lambda x:x['ref_id']):
        i=r['image_id']; info=refimages[i]; a=refanns[r['ann_id']]; sentence=r['sentences'][0]
        row=base(i,f'ref-{r["ref_id"]}-{sentence["sent_id"]}',info)
        row['question']=f'Locate the object referred to by: {sentence["sent"]}. Return only JSON {{"bbox":[x1,y1,x2,y2]}} on a 0-1000 scale.'
        x,y,w,h=a['bbox']; row['answer']={'bbox':[x,y,x+w,y+h]}
        row['metadata'].update(ref_id=r['ref_id'],sent_id=sentence['sent_id'],original_split=r['split'],answer_bbox_coordinate_system='pixel_xyxy',prediction_bbox_coordinate_system='qwen_0_1000')
        pools['refcoco_grounding'].append(row)
    for q in sorted(questions,key=lambda x:x['question_id']):
        i=q['image_id']; a=qab[q['question_id']]; row=base(i,q['question_id'],images[i])
        row.update(question=q['question']+' Answer with a short phrase only.',answer=a['multiple_choice_answer'])
        row['metadata']['answers']=[x['answer'] for x in a['answers']]
        pools['vqav2'].append(row)
    manifest={'seed':SEED,'counts':counts,'name':profile,'original_paper_sample_ids':'unconfirmed','split_rule':'global shuffled image IDs, 60/20/20; deterministic sampling; one image per task per split','sources':{'coco':'https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2014.zip','refcoco':'https://web.archive.org/web/20220413011718id_/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip','vqa':'https://visualqa.org/download.html'},'records':{}}
    selected=[]; tasks=[]
    for t,pool in pools.items():
        random.Random(SEED).shuffle(pool)
        splits={}; manifest['records'][t]={}
        for s,n in counts.items():
            rows=[]; used=set()
            for row in pool:
                i=row['metadata']['image_id']
                if splitby[i]==s and i not in used and not (s=='test' and i in excluded_test_images):
                    rows.append(row);used.add(i)
                if len(rows)==n:break
            assert len(rows)==n,(t,s,len(rows))
            if native_pixels and t in {'coco_detection','refcoco_grounding'}:
                for row in rows:
                    w=row['visible_context']['width'];h=row['visible_context']['height']
                    if t=='coco_detection':
                        label=row['metadata']['queried_category']
                        row['question']=f'Locate every {label} in the supplied image. Return only a JSON array of objects with label and bbox [x1,y1,x2,y2]. Use absolute pixel coordinates; image width is {w} and height is {h}.'
                    else:
                        expression=row['question'].split('Locate the object referred to by: ',1)[1].split('. Return only JSON',1)[0]
                        row['question']=f'Ground the referring expression: {expression}. Return only one JSON object with bbox [x1,y1,x2,y2]. Use absolute pixel coordinates; image width is {w} and height is {h}.'
                    row['metadata']['prediction_bbox_coordinate_system']='pixel_xyxy'
                    row['visible_context']['bbox_coordinate_system']='pixel_xyxy'
            path=ROOT/'code/data'/profile/t/f'{s}.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            splits[s]=str(path);selected.extend(rows)
            manifest['records'][t][s]=[{'id':r['id'],'image_id':r['metadata']['image_id']} for r in rows]
        splits['anchor']=splits['validation']
        tasks.append(dict(task_name=t,task_type='vqa' if t=='vqav2' else t,modality=['text','image'],splits=splits))
    sets={s:{x['image_id'] for t in manifest['records'].values() for x in t[s]} for s in counts}
    assert not (sets['train'] & sets['test'] or sets['validation'] & sets['test'] or sets['train'] & sets['validation'])
    manifest['cross_task_same_split_duplicates']={s:sum(len(t[s]) for t in manifest['records'].values())-len(sets[s]) for s in counts}
    manifest['excluded_previous_test_image_ids']=sorted(excluded_test_images)
    manifest['native_pixel_contract']=native_pixels
    manifest['anchor_policy']='runtime LLM selection from adaptation; manifest validation alias inactive in this profile'
    ROOT.joinpath('images').mkdir(exist_ok=True)
    def fetch(path):
        p=Path(path)
        if not p.exists():
            subset='train2014' if 'train2014' in p.name else 'val2014'
            url=f'https://s3.amazonaws.com/images.cocodataset.org/{subset}/{p.name}'
            with urllib.request.urlopen(url,timeout=60) as r: data=r.read(10_000_000)
            tmp=p.with_suffix('.part');tmp.write_bytes(data);tmp.replace(p)
        from PIL import Image
        with Image.open(p) as im: im.load()
    with ThreadPoolExecutor(max_workers=6) as ex:list(ex.map(fetch,sorted({r['image'] for r in selected})))
    ROOT.joinpath(f'code/configs/taskstream_{profile}.json').write_text(json.dumps({'tasks':tasks},indent=2))
    ROOT.joinpath('split_manifest.json' if profile=='local_image_disjoint' else f'split_manifest_{profile}.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'prepared':True,'images':len({r['image'] for r in selected}),'counts':counts}),flush=True)

if __name__=='__main__':build()
