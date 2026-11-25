"""Paired description study. No gold is available to the three prediction conditions."""
import json
import math
import os
import re
from collections import Counter
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from PIL import Image
from acquire import load_vlm
from panel import build

ROOT = Path(os.environ["RUN_ROOT"])
DESIGN = json.loads((ROOT/"design.json").read_text())
ARMS = ("direct","self","checked")
CONCLUSION = re.compile(r"\b(diagnos\w*|consistent with|suggest\w*|compatib\w* with|likely|typical (?:of|for)|characteristic of|indicative of|represents?|pathognomonic|suspicio\w* for|confirmed|differential|known case|history of|treated for)\b",re.I)


def norm(text):
    return " ".join(re.findall(r"[a-z0-9]+",text.casefold()))


def verify_span(source, span, options):
    if not span or len(span.split())>35 or len(span)<6:
        return "length"
    if span.casefold() not in source.casefold():
        return "not verbatim"
    if CONCLUSION.search(span):
        return "diagnostic conclusion"
    before = source.casefold().split(span.casefold(),1)[0]
    if re.search(r"\b(?:no|not|without|absence of|negative for|no evidence of)\s+(?:an?\s+|the\s+)?$",before):
        return "dropped negation"
    words = " "+norm(span)+" "
    for option in options:
        name=norm(option)
        acronym="".join(w[0] for w in name.split() if w not in {"of","the","and","with"})
        if " "+name+" " in words or len(acronym)>=3 and " "+acronym+" " in words:
            return "option text or acronym"
    return None


def paired(old,new,gold):
    if not (len(old)==len(new)==len(gold)):
        raise ValueError("unpaired predictions")
    gain = sum(a!=g and b==g for a,b,g in zip(old,new,gold))
    loss = sum(a==g and b!=g for a,b,g in zip(old,new,gold))
    d = gain+loss
    return dict(n=len(gold),reference=sum(a==g for a,g in zip(old,gold)),
                candidate=sum(b==g for b,g in zip(new,gold)),gain=gain,loss=loss,net=gain-loss,
                changed=sum(a!=b for a,b in zip(old,new)),
                exact_p=min(1.,2*sum(math.comb(d,i) for i in range(min(gain,loss)+1))/2**d))


def check_caption(caption, options, choose, generate):
    caption=" ".join(caption.split())
    attempts=[]
    if not caption:
        return "",[{"reason":"missing caption"}]
    banned=" | ".join(sorted(options))
    extract=("Copy ONE contiguous, verbatim span of at most 35 words from this caption that describes a visible "
             "abnormality or a meaningful negative finding. Preserve anatomy, negation and laterality. "
             "Remove conclusions, diagnostic names, synonyms, acronyms, eponyms, procedures performed and historical context. "
             "Do not output any named diagnosis, including these options: "+banned+
             ". If no safe descriptive span, output NONE. Return only the copied words. Caption:\n"+caption)
    for trial in range(2):
        if not trial and len(caption.split())<=35:
            span=caption
        else:
            hint="" if not attempts else "\nPrevious candidate rejected for "+attempts[-1]["reason"]+". Select a shorter safe span."
            span=generate(extract+hint,70).strip().strip('"“”')
        reason=verify_span(caption,span,options)
        if reason is None:
            support=choose("Source caption: "+caption+"\nCandidate wording: "+span,{},
                "Does the candidate preserve meaning including negation/laterality and describe a specific visible "
                "finding, rather than just the modality, acquisition or clinical history? A = yes, B = no.",["A","B"])
            leak=choose("Candidate wording: "+span+"\nDiagnostic options: "+banned,{},
                "A = reject: the candidate names a diagnosis, paraphrases a diagnostic option, contains its synonym, "
                "subtype, abbreviation, conclusion or clinical explanation. B = allow: only descriptive appearance, "
                "anatomy or meaningful negative findings, even if diagnostically useful. Choose A or B.",["A","B"])
            if support!="A" or leak!="B":
                reason="semantic disclosure or unsupported/nondescription"
        attempts.append({"span":span,"reason":reason or "accepted"})
        if reason is None:
            return span,attempts
    return "",attempts


def save(name,data):
    (ROOT/name).write_text(json.dumps(data,indent=2,ensure_ascii=False)+"\n")


def main():
    raw=Path(os.environ["RAW_ROOT"])
    cases,targets=build(raw,[Path(p) for p in os.environ["PRIOR_RUNS"].split(os.pathsep)])
    save("private_gold.json",targets)
    del targets
    choose_fn,generate_fn=load_vlm(with_text_generator=True)
    counts=Counter()
    def count(k):
        counts[k]+=1
        if sum(counts.values())>DESIGN["budget"]["max_total_model_calls"]:
            raise RuntimeError("call budget")
    def choose(ctx,opened,instruction,labels):
        count("image_decisions" if opened else "text_reviews")
        return choose_fn(ctx,opened,instruction,labels)
    def generate(prompt,tokens,image=None):
        count("image_descriptions" if image is not None else "text_extractions")
        return generate_fn(prompt,tokens,image)
    notes={}
    for i,(k,c) in enumerate(cases.items(),1):
        notes[k]=[]
        for im in c["images"]:
            text,attempts=check_caption(im["caption"],list(c["options"].values()),choose,generate)
            notes[k].append({"text":text,"audit":attempts})
        if i%16==0: print("reviewed",i,flush=True)
    save("private_checked.json",notes)
    outputs={}
    with ZipFile(raw/"images.zip") as z:
        for i,(k,c) in enumerate(cases.items(),1):
            images={}
            own=[]
            for j,v in enumerate(c["images"],1):
                with Image.open(BytesIO(z.read(v["path"]))) as im:
                    pix=im.convert("RGB")
                images[f"view{j}"]=pix
                text=generate("Describe the main visible abnormality or clinically relevant negative finding on this "
                              "single medical image, including its location and appearance if evident. Use one short "
                              "sentence, at most 35 words. No diagnosis, differential, history or speculation.",70,pix)
                own.append(" ".join(text.split()[:35]))
            ctx="Clinical history: "+c["history"]+"\nDiagnostic options:\n"+"\n".join(
                f"{a}. {s}" for a,s in c["options"].items())
            instruction="Using the images and history, choose the most likely diagnosis. Reply with its single option letter."
            pred={}
            checked=[v["text"] for v in notes[k]]
            for arm,fields in [("direct",[]),("self",own),("checked",checked)]:
                description="\n".join(f"View {j}: {v}" for j,v in enumerate(fields,1) if v)
                final=("Image observations:\n"+description+"\n" if description else "")+instruction
                pred[arm]=choose(ctx,images,final,list(c["options"]))
            if not any(checked) and pred["checked"]!=pred["direct"]:
                raise ValueError("identical image and prompt changed prediction")
            statuses=[]
            for ref,generated in zip(checked,own):
                if not ref:
                    statuses.append("unavailable")
                    continue
                answer=choose("Reference observation: "+ref+"\nModel-written observation: "+generated,{},
                    "Assess whether the primary specific finding and its relevant location are reported in the model "
                    "observation. A = covered including a synonymous description; B = omitted, too vague or not "
                    "enough detail; C = explicitly contradictory. Return one letter.",list("ABC"))
                statuses.append({"A":"covered","B":"omitted_or_insufficient","C":"contradicted"}[answer])
            outputs[k]=dict(pred=pred,stage=c["stage"],self_notes=own,checked=checked,coverage=statuses)
            if i%8==0: print("paired",i,flush=True)
    save("private_predictions.json",outputs)
    gold=json.loads((ROOT/"private_gold.json").read_text())
    def report(items):
        target=[gold[k] for k in items]
        def pred(arm): return [outputs[k]["pred"][arm] for k in items]
        return {b+"_vs_"+a:paired(pred(a),pred(b),target) for a,b in
                [("direct","checked"),("direct","self"),("self","checked")]}
    covered=[k for k,r in outputs.items() if any(r["checked"])]
    omissions={}
    for k,r in outputs.items():
        p=r["pred"]
        omissions[k]={"outcomes":{a:p[a]==gold[k] for a in ARMS},
                      "unreported_or_contradicted":[{"view":j,"reference":ref,"self":s,"status":status}
                      for j,(ref,s,status) in enumerate(zip(r["checked"],r["self_notes"],r["coverage"]),1)
                      if status in ("omitted_or_insufficient","contradicted")]}
    save("private_case_audit.json",omissions)
    transitions={}
    for k,r in outputs.items():
        status=("both_correct" if r["pred"]["direct"]==gold[k] else "rescued") if r["pred"]["checked"]==gold[k] else (
                "lost" if r["pred"]["direct"]==gold[k] else "both_wrong")
        bucket=transitions.setdefault(status,Counter())
        bucket["cases"]+=1
        bucket["cases_with_reported_omission_or_contradiction"]+=bool(omissions[k]["unreported_or_contradicted"])
    save("result.json",dict(all_cases=report(list(cases)),by_stage={s:report([k for k,r in outputs.items() if r["stage"]==s])
          for s in ("CT","nonCT")},available_cases=report(covered),cases_with_note=len(covered),
          notes_per_case=dict(Counter(sum(bool(x) for x in r["checked"]) for r in outputs.values())),
          per_image_coverage=dict(Counter(x for r in outputs.values() for x in r["coverage"])),
          direct_to_checked=transitions,physical_calls=counts))

if __name__=="__main__":
    main()
