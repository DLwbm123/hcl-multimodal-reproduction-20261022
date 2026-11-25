"""Bounded HCL and SkillOpt feedback updates on a frozen multi-image VQA agent."""
import json
import math
import os
import re
from pathlib import Path

from acquire import episode, load_vlm
from hcl.memory.abstract import AbstractMemory
from hcl.memory.raw import RawMemory
from patch_format import normalize
from upstream_skill import apply_patch_with_report
from panels import build

ROOT = Path(os.environ["RUN_ROOT"])
DESIGN = json.loads((ROOT / "design.json").read_text())
ARMS = DESIGN["arms"]
SEED = "Choose complementary views and use the observed findings and history to distinguish the diagnoses."


def save(name, value):
    (ROOT / (name + ".json")).write_text(json.dumps(value, indent=2) + "\n")


def state():
    return {"skill":DESIGN.get("seed_instruction",SEED), "memories":[]}


def words(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def guidance(st, case):
    text = st["skill"]
    if not st["memories"]:
        return text
    query = words(case["history"] + " " + " ".join(case["options"].values()) + " " +
                  " ".join(sorted({v["type"] for v in case["views"]})))
    def similarity(pair):
        tokens = words(pair[0]["question"])
        return len(query & tokens) / max(1,len(query | tokens))
    past, abstracts = max(((r,m["abstract"]) for m in st["memories"] for r in m["raw"]), key=similarity)
    notes = [r["guidance"] for r in abstracts if r.get("guidance")]
    return (text + "\nPrior training experience, useful only if applicable:\n" +
            past["question"][:650] + "\nVerified past diagnosis: " + str(past["correct_feedback"])[:160] +
            "\nTransferable guidance: " + " ".join(notes)[:560])


def compare(first, second, gold):
    if set(first) != set(second) or set(first) != set(gold):
        raise ValueError("Incomplete pairing")
    n = len(gold)
    a = sum(first[k]["answer"] == v for k,v in gold.items())
    b = sum(second[k]["answer"] == v for k,v in gold.items())
    gain = sum(first[k]["answer"] != v and second[k]["answer"] == v for k,v in gold.items())
    loss = gain - b + a
    d = gain+loss
    p = min(1., 2*sum(math.comb(d,k) for k in range(min(gain,loss)+1))/2**d)
    interval = None
    if d:
        x, z = gain/d, 1.959964
        mid = (x + z*z/(2*d))/(1+z*z/d)
        radius = z*math.sqrt(x*(1-x)/d + z*z/(4*d*d))/(1+z*z/d)
        interval = [mid-radius,mid+radius]
    extra = {}
    if first and "actions" in next(iter(first.values())):
        extra["changed_tool_sequence"] = sum(first[k]["actions"] != second[k]["actions"] for k in gold)
        extra["changed_action_or_argument"] = sum(first[k]["trace"] != second[k]["trace"] for k in gold)
        from collections import Counter
        extra["candidate_tool_counts"] = dict(Counter(a for x in second.values() for a in x["actions"]))
    return dict(**extra,n=n,reference=a,candidate=b,gain=gain,loss=loss,net=b-a,exact_p=p,
                discordant_win_fraction_95ci=interval,
                changed_diagnoses=sum(first[k]["answer"]!=second[k]["answer"] for k in gold),
                changed_image_sets=sum(set(first[k]["views"])!=set(second[k]["views"]) for k in gold))


def evidence(cases, outputs, gold):
    items = []
    for c in cases:
        k = c["case"]
        desc = {v["view"]:v["type"] for v in c["views"]}
        observed = [desc[v] for v in outputs[k]["views"]]
        question = ("History: " + c["history"][:350] + "\nOpened: " + ", ".join(observed) +
                    "\nDifferential: " + " | ".join(c["options"].values())[:380])
        if "trace" in outputs[k]:
            question = question[:500] + "\nActions: " + " -> ".join(outputs[k]["trace"])
        items.append(dict(task_id=k, question=question, response=c["options"][outputs[k]["answer"]],
                          answer=c["options"][gold[k]], split="fit", task_type="diagnosis"))
    return items


def update_memory(st, stage, examples, directory, generator):
    raw = RawMemory(directory / "raw.jsonl", capacity_per_task=4)
    raw.add(stage,examples,[dict(task_id=x["task_id"],answer=x["response"]) for x in examples])
    class Adapter:
        def generate(self,prompt,*,state):
            self.output = generator(prompt,state["max_new_tokens"])
            return self.output
    adapter = Adapter()
    abstract = AbstractMemory(directory / "abstract.json",model=adapter,mode="llm",max_new_tokens=384)
    abstract.consolidate(stage,examples,raw_records=raw.records)
    (directory / "private_generated.txt").write_text(adapter.output)
    return [*st["memories"],{"raw":raw.retrieval_candidates(),"abstract":abstract.retrieval_candidates()}]


def edit_skill(current, examples, memories, gen):
    payload = [{"case_context":r["question"][:500],"model_diagnosis":r["response"],
                "verified_diagnosis":r["answer"]} for r in examples]
    prompt = ("Improve the reusable procedure for " + DESIGN.get("procedure", "choosing 3 views and comparing diagnostic hypotheses") + ". "
              "Reflect on recurring successes and errors in training and propose one short transferable "
              "instruction. Do not name training diagnoses, labels or cases. Return only JSON of the form "
              '{"edits":[{"op":"append","content":"one actionable instruction"}]}. '
              'Alternatively replace the exact current instruction. Skill must stay under 700 characters. '
              "\nCurrent skill: " + current + "\nTraining trajectories: " + json.dumps(payload))
    if memories:
        prompt += "\nCandidate abstract guidance: " + json.dumps(memories[-1]["abstract"])
    attempts = []
    for _ in range(2):
        output = gen(prompt,256)
        try:
            patch, fmt = normalize(output,current)
            updated, reports = apply_patch_with_report(current,patch)
            if reports[0]["status"] not in {"applied_append","applied_replace"} or updated==current or len(updated)>700:
                raise ValueError("Unapplied, unchanged or overlong edit")
            attempts.append(dict(output=output,format=fmt,status=reports[0]["status"]))
            return updated, attempts
        except (IndexError,KeyError,ValueError,TypeError) as exc:
            attempts.append(dict(output=output,error=str(exc)))
            prompt += "\nCorrect this malformed or overlong edit: " + output + "\nError: " + str(exc)
    return None, attempts


def main():
    panels = build(Path(os.environ["RAW_ROOT"]),ROOT,DESIGN)
    choose, generate = load_vlm(with_text_generator=True)
    use_multiple_tools = DESIGN.get("multi_tool",False)
    if use_multiple_tools:
        from visual_ops import multi_episode
    saved = {}
    counts = {"decisions":0,"generations":0}
    def count(kind):
        counts[kind] += 1
        if sum(counts.values()) > DESIGN["budget"]["max_total_model_calls"]:
            raise RuntimeError("call budget exhausted")
    def gen(prompt,tokens):
        count("generations")
        return generate(prompt,tokens)
    def predict(st,stage,role,fixed=False):
        directory,cases = panels[stage,role]
        results = {}
        for case in cases:
            hint = guidance(st,case)
            key = (case["case"],hint,fixed)
            if key not in saved:
                def guided(context,opened,instruction,labels):
                    count("decisions")
                    return choose(context,opened,"Procedure:\n"+hint+"\nCurrent action:\n"+instruction,labels)
                if use_multiple_tools:
                    saved[key] = multi_episode(guided,directory,case,fixed)
                    results[case["case"]] = saved[key]
                    continue
                output = episode(guided,directory,case,True)
                opened = [r["returned"] for r in output["receipts"]]
                if len(set(opened)) != 3 or output["answer"] not in case["options"]:
                    raise ValueError("invalid rollout")
                saved[key] = dict(answer=output["answer"],views=opened)
            results[case["case"]] = saved[key]
        return results
    def gold(stage,role):
        return json.loads((panels[stage,role][0]/"gold.json").read_text())
    states = {arm:state() for arm in ARMS}
    events, proposals, after_a = [], {}, {}
    for stage in ("A","B"):
        for arm in ARMS[1:]:
            incumbent = states[arm]
            feedback = evidence(panels[stage,"fit"][1],predict(incumbent,stage,"fit"),gold(stage,"fit"))
            cand = dict(incumbent)
            folder = ROOT / (arm+"_"+stage)
            folder.mkdir()
            if arm in {"hcl","hybrid"}:
                cand["memories"] = update_memory(incumbent,stage,feedback,folder,gen)
            valid = True
            if arm in {"skillopt","hybrid"}:
                skill,attempts = edit_skill(incumbent["skill"],feedback,cand["memories"],gen)
                (folder/"private_patches.json").write_text(json.dumps(attempts))
                valid = skill is not None
                if valid:
                    cand["skill"] = skill
                else:
                    cand = incumbent
            def gate(which):
                return compare(predict(incumbent,which,"gate"),predict(cand,which,"gate"),gold(which,"gate"))
            current = gate(stage)
            old = gate("A") if stage=="B" and arm in {"hybrid","hcl"} else None
            accepted = valid and current["net"]>0 and (old is None or old["net"]>=0)
            if accepted:
                states[arm] = cand
            proposals[arm] = cand
            events.append(dict(stage=stage,arm=arm,valid=valid,accepted=accepted,current=current,old=old))
            save("gates",events)
            print("gate",stage,arm,"valid",valid,"accepted",accepted,flush=True)
        if stage=="A":
            after_a = dict(states)
    save("private_frozen_states",{"deployed":states,"last_proposed":proposals,"after_A":after_a})
    preds, candidate_preds, previous = {}, {}, {}
    for stage in ("A","B"):
        preds[stage],candidate_preds[stage] = {},{}
        for arm in ARMS:
            preds[stage][arm] = predict(states[arm],stage,"confirm")
            if arm != "base":
                candidate_preds[stage][arm] = predict(proposals[arm],stage,"confirm")
                if stage=="A":
                    previous[arm] = predict(after_a[arm],stage,"confirm")
            print("confirmation",stage,arm,flush=True)
    fixed_outputs = {}
    if use_multiple_tools:
        for stage in ("A","B"):
            fixed_outputs.update(predict(state(),stage,"confirm",fixed=True))
    save("private_confirmation",dict(deployed=preds,last_proposed=candidate_preds,after_A=previous,
                                     fixed_schedule=fixed_outputs))
    truths = {s:gold(s,"confirm") for s in ("A","B")}
    all_gold = {**truths["A"],**truths["B"]}
    def pooled(table,arm):
        return {**table["A"][arm],**table["B"][arm]}
    ref = pooled(preds,"base")
    final = {arm:compare(ref,pooled(preds,arm),all_gold) for arm in ARMS}
    diagnostic = {arm:compare(ref,pooled(candidate_preds,arm),all_gold) for arm in ARMS[1:]}
    retention = {arm:compare(previous[arm],preds["A"][arm],truths["A"]) for arm in ARMS[1:]}
    by_stage = {s:{arm:compare(preds[s]["base"],preds[s][arm],truths[s]) for arm in ARMS} for s in truths}
    extra = {"fixed_schedule":compare(ref,fixed_outputs,all_gold)} if use_multiple_tools else {}
    save("result",dict(**extra,deployed=final,stage_results=by_stage,last_proposed=diagnostic,
                       fresh_A_retention=retention,gate_decisions=events,physical_calls=counts))


if __name__=="__main__":
    main()
