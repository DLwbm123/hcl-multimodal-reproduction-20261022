import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageChops, ImageDraw, ImageFilter

ROOT=Path(os.environ["RUN_ROOT"])
BASE=Path(os.environ["PILOT_ROOT"])
OUT=ROOT/"run"
P=json.loads((ROOT/"protocol.json").read_text())
B=json.loads((ROOT/"budget.json").read_text())
sys.path[:0]=[str(ROOT),str(BASE/"code"),*P["python_paths"]]
from runtime import FrozenModels, EpisodeTools
from evaluate import score
from routing import evaluate, fit, summarize


def guard():
    stamp=json.loads((ROOT/"resource_status.json").read_text())["time"]
    if time.time()>=B["soft_deadline_epoch"] or time.time()-stamp>120 or (ROOT/"HARD_STOP.json").exists():
        raise RuntimeError("resource guard")


def proposals(models,image,box):
    # Same SAM preprocessing, multimask inference and native-size restoration as FrozenModels.segment.
    guard()
    models.sam_calls+=1
    inputs=models.processor(images=image,input_boxes=[[box]],return_tensors="pt").to("cuda")
    with torch.inference_mode():
        pred=models.sam(**inputs,multimask_output=True)
    masks=models.processor.image_processor.post_process_masks(pred.pred_masks.cpu(),inputs["original_sizes"].cpu(),inputs["reshaped_input_sizes"].cpu())[0][0]
    values=pred.iou_scores[0,0].detach().float().cpu().numpy()
    assert len(masks)==len(values)==3 and np.isfinite(values).all()
    return masks.numpy().astype(bool),values


def panel(image,masks,order,path):
    preview=image.copy()
    preview.thumbnail((384,384))
    w,h=preview.size
    canvas=Image.new("RGB",(2*w,2*(h+24)),"white")
    draw=ImageDraw.Draw(canvas)
    for idx in range(4):
        tile=preview.copy()
        if idx:
            mask=Image.fromarray(masks[order[idx-1]].astype("uint8")*255).resize((w,h),Image.Resampling.NEAREST)
            inner=mask.filter(ImageFilter.MinFilter(3))
            tile.paste((0,255,255), mask=ImageChops.subtract(mask,inner))
        x,y=idx%2*w,idx//2*(h+24)
        canvas.paste(tile,(x,y+24))
        draw.text((x+4,y+4),("Original","A","B","C")[idx],fill="black")
    canvas.save(path)


def run_set(models,cases,role):
    rows=[]
    rng=random.Random(20261125 if role=="train" else 20261126)
    for i,case in enumerate(cases):
        guard()
        order=list(range(3));rng.shuffle(order)
        models.phase,models.scope=role,str(i)
        tools=EpisodeTools(models,case["image"],OUT/"private_artifacts"/role/str(i),P["runtime"])
        loc=tools.execute("localize",{})
        zero={"score":0.0,"valid":False,"success":False}
        row={"available":False,"default":zero,"review":zero,"oracle":0.0,"switched":False,"parse":"no_box"}
        if loc["status"]=="ok":
            masks,values=proposals(models,tools.image,loc["bbox"])
            assert masks.shape==(3,tools.h,tools.w)
            ranks=np.argsort(-values,kind="stable")
            top,second=map(int,ranks[:2])
            path=tools.out/"candidates.png";panel(tools.image,masks,order,path)
            prompt=("Upper left is the unmarked current endoscopy image. The other three panels show candidate polyp mask outlines in cyan. "
                    "Choose the one that follows the visible lesion boundary best, avoiding unrelated tissue or missing lesion area. "
                    "Reply with exactly A, B, C, or UNSURE if the evidence is insufficient.")
            raw=models.generate(prompt,str(path),32)
            label=raw.strip().upper().strip(" .\"'`\n")
            choice=order["ABC".index(label)] if label in ("A","B","C") else top
            # Gold evaluation happens only after the blind reviewer has committed its choice.
            measured=[]
            for j,mask in enumerate(masks):
                key=str(j);tools.masks[key]=mask
                ans={"mask_id":key,"area_fraction":float(mask.mean())}
                measured.append(score("T2",case,{"answer":ans},tools,P["evaluation"]))
            union=(masks[top]|masks[second]).sum()
            agreement=float((masks[top]&masks[second]).sum()/union) if union else 1.
            row.update(available=True,default=measured[top],review=measured[choice],oracle=max(v["score"] for v in measured),
                       switched=choice!=top,margin=float(values[top]-values[second]),agreement=agreement,
                       parse="choice" if label in ("A","B","C") else "abstain" if label=="UNSURE" else "invalid")
        rows.append(row)
        with (OUT/("private_"+role+".jsonl")).open("a") as f:
            f.write(json.dumps({"id":case["id"],**row})+"\n")
    models.assert_frozen()
    return rows


def main():
    data=json.loads((BASE/"private_data.json").read_text())
    train,dev=data["train"],data["validation"]
    assert len(train)==len(dev)==12 and len({c["group_id"] for c in train+dev})==24
    models=FrozenModels(P,OUT,guard)
    rows=run_set(models,train,"train")
    t=summarize(rows);policy=fit(rows);t["frozen_policy"]=policy
    loo=[fit(rows[:i]+rows[i+1:]) for i in range(len(rows))]
    t["loo_metrics"]=evaluate(rows,loo)
    t["open_development"]=bool(t["material_gain"] and t["material_loss"] and t["headroom"]>=.02)
    (OUT/"training_decision.json").write_text(json.dumps(t,indent=2))
    result={"training":t,"development":None}
    if t["open_development"]:
        d=summarize(run_set(models,dev,"development"))
        m=d["policies"][policy];fixed=[d["policies"][p] for p in ("default","review")]
        d["frozen_meets_criterion"]=0<m["reviews"]<len(dev) and m["dice"]>=max(a["dice"] for a in fixed)+.02 and m["success"]>=max(a["success"] for a in fixed)
        result["development"]=d
    result.update(vlm_calls=models.calls,sam_calls=models.sam_calls)
    (OUT/"result.json").write_text(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
