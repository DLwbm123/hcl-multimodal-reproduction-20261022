"""Frozen Qwen2.5-VL probe for whether one additional endoscopy view helps."""
from collections import Counter
from contextlib import ExitStack
import io
import json
import os
from pathlib import Path
import random
import statistics
import time
from zipfile import ZipFile

from PIL import Image, ImageOps

N = 48
SEED = 26112804
GRID = ('grid_tl', 'grid_tr', 'grid_bl', 'grid_br')
ACTIONS = ('direct', 'repeat', *GRID)
DEPLOY = ('direct', *GRID)
QUESTION = 'Question: Is there a colorectal polyp visible in this endoscopy frame? Answer Yes or No.'


def load(archive, name, mode):
    with Image.open(io.BytesIO(archive.read(name))) as im:
        return im.convert(mode)


def fingerprint(image):
    small = ImageOps.grayscale(image)
    horizontal = list(small.resize((9, 8), Image.Resampling.LANCZOS).getdata())
    vertical = list(small.resize((8, 9), Image.Resampling.LANCZOS).getdata())
    bits = [horizontal[y*9+x] > horizontal[y*9+x+1] for y in range(8) for x in range(8)]
    bits += [vertical[y*8+x] > vertical[(y+1)*8+x] for y in range(8) for x in range(8)]
    return int(''.join('1' if v else '0' for v in bits), 2)


def panel(frames, annotated):
    members = [Path(x) for x in frames.namelist() if Path(x).suffix.lower() == '.jpg']
    masks = {Path(x).stem: x for x in annotated.namelist()
             if Path(x).parent.name == 'masks' and Path(x).suffix.lower() in ('.jpg', '.png')}
    references = {Path(x).stem: x for x in annotated.namelist()
                  if Path(x).parent.name == 'images' and Path(x).suffix.lower() == '.jpg'}
    positives = [n for n in members if n.parent.name == 'polyps' and n.stem in masks and n.stem in references]
    pools = [('polyps', N, positives)]
    for name in ('cecum', 'retroflex-rectum'):
        pools.append((name, N//2, [n for n in members if n.parent.name == name]))
    rng = random.Random(SEED)
    seen = []
    rows = []
    for kind, count, pool in pools:
        pool = sorted(pool)
        rng.shuffle(pool)
        selected = 0
        for name in pool:
            image = load(frames, str(name), 'RGB')
            digest = fingerprint(image)
            if any((digest ^ prior).bit_count() <= 6 for prior in seen):
                continue
            mask_name, ref_name = masks.get(name.stem), references.get(name.stem)
            if kind == 'polyps':
                reference = load(annotated, ref_name, 'RGB')
                mask = load(annotated, mask_name, 'L')
                if (mask.size != reference.size or (digest ^ fingerprint(reference)).bit_count() > 12
                    or abs(image.width/image.height-reference.width/reference.height) > .02
                    or not mask.point(lambda x: 255 if x >= 128 else 0).getbbox()):
                    continue
            else:
                assert mask_name is None and ref_name is None
            seen.append(digest)
            rows.append(dict(member=str(name), mask=mask_name, label=(kind=='polyps'), category=kind))
            selected += 1
            if selected == count:
                break
        assert selected == count, (kind, selected)
    rng.shuffle(rows)
    return rows


def views(image, mask):
    w,h=image.size
    cw,ch=round(.62*w),round(.62*h)
    assert cw >= 28 and ch >= 28
    grid = ((0,0,cw,ch),(w-cw,0,w,ch),(0,h-ch,cw,h),(w-cw,h-ch,w,h))
    result = {'direct':[image], 'repeat':[image,image]}
    result.update({action:[image,image.crop(box)] for action,box in zip(GRID,grid,strict=True)})
    if mask is not None:
        mask=mask.resize(image.size,Image.Resampling.NEAREST).point(lambda x:255 if x>=128 else 0)
        x1,y1,x2,y2=mask.getbbox()
        x=min(max(round((x1+x2-cw)/2),0),w-cw)
        y=min(max(round((y1+y2-ch)/2),0),h-ch)
        lesion=(x,y,x+cw,y+ch)
        result['lesion']=[image,image.crop(lesion)]
        area=mask.histogram()[255]
        coverage=[mask.crop(b).histogram()[255]/area for b in (*grid,lesion)]
        best=max(range(len(GRID)),key=coverage.__getitem__)
        return result,best,coverage
    return result,None,None


class Reader:
    def __init__(self):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.torch = torch
        base = os.environ['MODEL']
        self.processor = AutoProcessor.from_pretrained(base, local_files_only=True,
                          min_pixels=56*56, max_pixels=448*448)
        self.processor.tokenizer.padding_side='left'
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(base,
                     local_files_only=True, torch_dtype=torch.bfloat16,
                     attn_implementation='sdpa', device_map={'':0}).eval().requires_grad_(False)
        self.prompts = {}
        tokenizer=self.processor.tokenizer
        for n in (1,2):
            note = ('The image is the full endoscopy frame.' if n==1 else
                    'The first image is the full endoscopy frame. The second image is an additional view of the same frame.')
            content=[{'type':'image'} for _ in range(n)]+[{'type':'text','text':note+' '+QUESTION}]
            self.prompts[n]=self.processor.apply_chat_template([{'role':'user','content':content}],
                                                              tokenize=False,add_generation_prompt=True)
        self.answer_ids = [tokenizer.encode(word,add_special_tokens=False) for word in ('No','Yes')]
        assert all(len(v)==1 for v in self.answer_ids)
        for p in self.prompts.values():
            ids=tokenizer.encode(p,add_special_tokens=False)
            assert all(tokenizer.encode(p+w,add_special_tokens=False)==ids+t
                       for w,t in zip(('No','Yes'),self.answer_ids,strict=True))
        self.answer_ids=[v[0] for v in self.answer_ids]

    def score(self,batch):
        texts=[self.prompts[len(images)] for _,images in batch]
        flat=[image for _,images in batch for image in images]
        inputs=self.processor(text=texts,images=flat,padding=True,return_tensors='pt').to('cuda')
        assert inputs.attention_mask[:,-1].all()
        with self.torch.inference_mode():
            logits=self.model(**inputs,use_cache=False).logits[:,-1,self.answer_ids].float()
            probability=logits.softmax(-1)[:,1]
        assert probability.isfinite().all()
        return {name:float(p) for (name,_),p in zip(batch,probability.cpu(),strict=True)}


def predicted(row, key):
    return row['scores'][key] > .5


def counts(rows,key):
    pairs=[(row['label'],predicted(row,key)) for row in rows]
    return dict(tp=sum(y and p for y,p in pairs),fn=sum(y and not p for y,p in pairs),
                tn=sum(not y and not p for y,p in pairs),fp=sum(not y and p for y,p in pairs))


def paired(rows, reference, candidate):
    prior=[predicted(r,reference)==r['label'] for r in rows]
    after=[predicted(r,candidate)==r['label'] for r in rows]
    return dict(rescue=sum(b and not a for a,b in zip(prior,after)),
                harm=sum(a and not b for a,b in zip(prior,after)), correct=sum(after), n=len(rows))


def aggregate(rows):
    positives=[r for r in rows if r['label']]
    negatives=[r for r in rows if not r['label']]
    fixed={k:counts(rows,k) for k in ACTIONS}
    best=max(DEPLOY,key=lambda k:(fixed[k]['tp']+fixed[k]['tn'],-int(k!='direct')))
    oracle=[]
    for r in rows:
        action=next((k for k in DEPLOY if predicted(r,k)==r['label']), 'direct')
        oracle.append({**r,'scores':{**r['scores'],'oracle':r['scores'][action]},'oracle_action':action})
    upper=counts(oracle,'oracle')
    delta=upper['tp']+upper['tn']-fixed[best]['tp']-fixed[best]['tn']
    lesion={ref:paired(positives,ref,'lesion') for ref in ('direct','repeat')}
    joint_rescues=sum(predicted(r,'lesion') and not predicted(r,'direct') and not predicted(r,'repeat')
                      for r in positives)
    return dict(n=len(rows),positive=len(positives),negative_proxy=len(negatives),
                by_class=dict(Counter(r['category'] for r in rows)),fixed=fixed,
                best_fixed=best,best_fixed_reviews=len(rows)*int(best!='direct'),
                oracle=upper,oracle_reviews=sum(r['oracle_action']!='direct' for r in oracle),
                oracle_delta_accuracy_pp=100*delta/len(rows),
                lesion_positive=lesion,lesion_joint_rescues=joint_rescues,
                mask_grid_positive={ref:paired(positives,ref,'best_mask_grid') for ref in ('direct','repeat')},
                diagnostic_mask_coverage=dict(lesion_median=statistics.median(r['coverage'][-1] for r in positives),
                  best_grid_median=statistics.median(max(r['coverage'][:-1]) for r in positives)),
                gate_lesion_joint_rescues_ge_3=joint_rescues>=3,
                gate_oracle_at_least_5_pp=100*delta/len(rows)>=5)


def main():
    start=time.monotonic()
    root=Path(os.environ['DATA_ROOT'])
    with ExitStack() as ctx:
        frames=ctx.enter_context(ZipFile(root/'hyper-kvasir-labeled-images.zip'))
        annotated=ctx.enter_context(ZipFile(root/'hyper-kvasir-segmented-images.zip'))
        samples=panel(frames,annotated)
        Path('private_panel.json').write_text(json.dumps(samples))
        print('panel frozen', len(samples),flush=True)
        reader=Reader()
        rows=[]
        with open('private_predictions.jsonl','w') as out:
            for i,row in enumerate(samples):
                image=load(frames,row['member'],'RGB')
                mask=load(annotated,row['mask'],'L') if row['mask'] else None
                images,best,coverage=views(image,mask)
                actions=list(images.items())
                scores={}
                for j in range(0,len(actions),4):
                    scores.update(reader.score(actions[j:j+4]))
                if best is not None:
                    scores['best_mask_grid']=scores[GRID[best]]
                row={**row,'scores':scores, 'coverage':coverage}
                rows.append(row)
                out.write(json.dumps(row)+'\n')
                if i==0 or (i+1)%12==0:
                    out.flush()
                    print('completed',i+1,'seconds',round(time.monotonic()-start,1),flush=True)
        summary=aggregate(rows)
        summary.update(seconds=round(time.monotonic()-start,3),max_reserved_mib=reader.torch.cuda.max_memory_reserved()/2**20)
        Path('result.tmp').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
        Path('result.tmp').replace('result.json')
        print('finished',flush=True)


if __name__=='__main__':
    main()
