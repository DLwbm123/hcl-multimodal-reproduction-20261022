import json
import os
from collections import Counter
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from PIL import Image
from acquire import load_vlm
from panel import build
from review import check_caption, paired
from previous_learning import update_memory
from patch_format import normalize
from upstream_skill import apply_patch_with_report

ROOT=Path(os.environ["RUN_ROOT"])
OLD=Path(os.environ["FINDINGS_RUN"])
BASE=("Describe the main visible abnormality or clinically relevant negative finding on this "
      "single medical image, including its location and appearance if evident. Use one short "
      "sentence, at most 35 words. No diagnosis, differential, history or speculation.")
JUDGE=("Assess whether the primary specific finding and its relevant location are reported in the model "
       "observation. A = covered including a synonymous description; B = omitted, too vague or not "
       "enough detail; C = explicitly contradictory. Return one letter.")
def save(name,x):
    (ROOT/(name+".json")).write_text(json.dumps(x,indent=2)+"\n")


def propose(cases, gen):
    memory=[]
    for stage,rows in cases.items():
        examples=[]
        for k,r in rows:
            for j,(note,ref,status) in enumerate(zip(r["self_notes"],r["checked"],r["coverage"])):
                if not ref:
                    continue
                examples.append(dict(task_id=f"{k}_{j}",split="fit",task_type="finding_coverage",
                    question="Observe this image. Earlier visual report: "+note+"\nVerified descriptive finding: "+ref+
                             "\nWas the main reference finding covered by the report?",
                    response="yes" if status=="covered" else "no", answer="yes"))
        folder=ROOT/("memory_"+stage)
        folder.mkdir()
        memory=update_memory({"memories":memory},stage,examples,folder,gen)
    save("private_memory",memory)
    evidence=[dict(abstract=m["abstract"],cases=[r["question"] for r in m["raw"]]) for m in memory]
    prompt=("Edit a reusable medical image observation skill after reviewing feedback on missed findings. "
            "Append one concrete short inspection procedure that a visual model can execute on the next image, "
            "including which visual features to examine and what to report if visible. No disease names, "
            "patient-specific findings, fixed anatomic location, instructions to obtain history, diagnosis or "
            "treatment. Keep output at one sentence of 35 words per examined image. Respond with JSON only: "
            '{"edits":[{"op":"append","content":"your short reusable instruction"}]}. '
            "No path, target or extra keys for append.\nCurrent skill: "+BASE+"\nFeedback: "+json.dumps(evidence))
    attempts=[]
    for _ in range(2):
        output=gen(prompt,200)
        try:
            patch,_=normalize(output,BASE)
            if patch["edits"][0]["op"]!="append":
                raise ValueError("Only append is permitted")
            skill,reports=apply_patch_with_report(BASE,patch)
            if reports[0]["status"]!="applied_append" or skill==BASE or len(skill)>700:
                raise ValueError("Not applied or too long")
            attempts.append(dict(output=output,status="applied_append"))
            save("private_patch",attempts)
            return skill
        except (ValueError,KeyError,IndexError,TypeError) as e:
            attempts.append(dict(output=output,error=str(e)))
            prompt+="\nInvalid previous answer: "+output+"\nRepair its JSON. "+str(e)
    save("private_patch",attempts)
    return BASE


def main():
    raw=Path(os.environ["RAW_ROOT"])
    old=json.loads((OLD/"private_predictions.json").read_text())
    dirs=[Path(s) for s in os.environ["PRIOR_RUNS"].split(os.pathsep)]
    prior,prior_gold=build(raw,dirs)
    if set(prior)!=set(old):raise ValueError("unexpected prior panel")
    fresh,gold=build(raw,dirs,exclude=prior)
    if set(prior)&set(fresh) or len(fresh)!=64:raise ValueError("fresh panel invalid")
    (ROOT/"private_targets.json").write_text(json.dumps(gold))
    del gold
    choose_fn,gen_fn=load_vlm(with_text_generator=True)
    design=json.loads((ROOT/"design.json").read_text())
    calls=Counter()
    def count(key):
        calls[key]+=1
        if sum(calls.values())>design["budget"]["max_total_model_calls"]:raise RuntimeError("call limit")
    def choose(ctx,images,prompt,labels):
        count("image_decisions" if images else "text_reviews")
        return choose_fn(ctx,images,prompt,labels)
    def gen(prompt,tokens,image=None):
        count("image_notes" if image is not None else "text_generations")
        return gen_fn(prompt,tokens,image)
    fit={}
    gate_ids=[]
    for s in ["CT","nonCT"]:
        rows=[(k,r) for k,r in old.items() if r["stage"]==s]
        fit[s]=rows[:16]
        gate_ids.extend(k for k,r in rows[16:])
    skill=propose(fit,gen)
    save("private_candidate",skill)
    print("patch applied",skill!=BASE,flush=True)
    def examine(c,images,procedure,refs):
        own=[" ".join(gen(procedure,70,img).split()[:35]) for img in images.values()]
        ctx="Clinical history: "+c["history"]+"\nDiagnostic options:\n"+"\n".join(f"{k}. {v}" for k,v in c["options"].items())
        desc="\n".join(f"View {j}: {text}" for j,text in enumerate(own,1) if text)
        instruction=("Image observations:\n"+desc+"\n" if desc else "")+"Using the images and history, choose the most likely diagnosis. Reply with its single option letter."
        answer=choose(ctx,images,instruction,list(c["options"]))
        status=[]
        for ref,note in zip(refs,own):
            if not ref:
                status.append("unavailable")
            else:
                label=choose("Reference observation: "+ref+"\nModel-written observation: "+note,{},JUDGE,list("ABC"))
                status.append(dict(A="covered",B="omitted_or_insufficient",C="contradicted")[label])
        return dict(answer=answer,status=status,notes=own)
    def images(c,archive):
        imgs={}
        for j,v in enumerate(c["images"],1):
            with Image.open(BytesIO(archive.read(v["path"]))) as src: imgs[f"view{j}"]=src.convert("RGB")
        return imgs
    def metrics(rows,truth):
        scores=paired([r["seed"]["answer"] for r in rows.values()],[r["candidate"]["answer"] for r in rows.values()],
                      [truth[k] for k in rows])
        counts={arm:Counter(s for r in rows.values() for s in r[arm]["status"]) for arm in ["seed","candidate"]}
        scores.update(coverage=counts,covered_net=counts["candidate"]["covered"]-counts["seed"]["covered"])
        return scores
    with ZipFile(raw/"images.zip") as archive:
        gate={}
        for k in gate_ids:
            row=old[k]
            seed=dict(answer=row["pred"]["self"],status=row["coverage"],notes=row["self_notes"])
            candidate=examine(prior[k],images(prior[k],archive),skill,row["checked"]) if skill!=BASE else seed
            gate[k]=dict(stage=row["stage"],seed=seed,candidate=candidate)
        gate_metrics={s:metrics({k:r for k,r in gate.items() if r["stage"]==s},prior_gold) for s in fit}
        accepted=skill!=BASE and all(r["net"]>=0 for r in gate_metrics.values()) and sum(r["covered_net"] for r in gate_metrics.values())>0
        save("gate",dict(accepted=accepted,by_stage=gate_metrics))
        save("private_gate",gate)
        print("gate frozen",accepted,flush=True)
        rows={}
        for i,(k,c) in enumerate(fresh.items(),1):
            imgs=images(c,archive)
            refs=[check_caption(v["caption"],list(c["options"].values()),choose,gen)[0] for v in c["images"]]
            seed=examine(c,imgs,BASE,refs)
            candidate=examine(c,imgs,skill,refs) if skill!=BASE else seed
            rows[k]=dict(stage=c["stage"],refs=refs,seed=seed,candidate=candidate)
            if i%8==0:print("fresh",i,flush=True)
    save("private_predictions",rows)
    gold=json.loads((ROOT/"private_targets.json").read_text())
    candidate=metrics(rows,gold)
    deployed=candidate if accepted else metrics({k:dict(seed=r["seed"],candidate=r["seed"]) for k,r in rows.items()},gold)
    save("result",dict(n=64,accepted=accepted,valid_patch=skill!=BASE,gate=gate_metrics,deployed=deployed,
                       candidate=candidate,by_stage={s:metrics({k:r for k,r in rows.items() if r["stage"]==s},gold) for s in fit},
                       calls=calls))

if __name__=="__main__":
    main()
