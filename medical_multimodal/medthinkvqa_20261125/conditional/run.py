import json
import os
import random
import re
from collections import Counter
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from PIL import Image
from acquire import load_vlm
from panel import build
from previous_learning import update_memory
from prior_skill import BASE, JUDGE
from review import check_caption, paired
from upstream_skill import apply_patch_with_report

ROOT = Path(os.environ["RUN_ROOT"])
DESIGN = json.loads((ROOT / "design.json").read_text())
TARGET = ("Use the clinical history and the complete list of differential diagnoses to guide inspection "
          "of this single image. Report one directly visible finding or visible negative that helps "
          "distinguish the alternatives, with location and appearance if evident. Do not name a "
          "diagnosis, repeat history, or infer invisible findings. One sentence, at most 35 words.")
AUDIT = ("Audit the observation against these pixels. Does it assert any visible finding, location, "
         "or absence for which the displayed image provides insufficient evidence? A=yes, B=no.")
calls = Counter()


def save(name, value):
    (ROOT / (name + ".json")).write_text(json.dumps(value, indent=2) + "\n")


def measured(fn, role):
    def wrapped(*args):
        calls[role] += 1
        if sum(calls.values()) > DESIGN["budget"]["max_total_model_calls"]:
            raise RuntimeError("call budget exhausted")
        return fn(*args)
    return wrapped


def images(case, archive):
    result = {}
    for i, v in enumerate(case["images"], 1):
        with Image.open(BytesIO(archive.read(v["path"]))) as src:
            result[f"view{i}"] = src.convert("RGB")
    assert len(result) == 3
    return result


def context(c):
    return "Clinical history: " + c["history"] + "\nDiagnostic options:\n" + "\n".join(
        f"{k}. {v}" for k, v in c["options"].items())


def grade(ref, note, choose):
    if not ref:
        return "-"
    return choose("Reference observation: " + ref + "\nModel-written observation: " + note, {}, JUDGE, list("ABC"))


def write_notes(prompt, imgs, gen):
    return [" ".join(gen(prompt, 70, img).split()[:35]) for img in imgs.values()]


def proposals(training, gen):
    memory, feedback = [], {}
    for stage in ("CT", "nonCT"):
        examples = []
        for alias, row in training.items():
            if row["stage"] != stage or not row["ref"]:
                continue
            question = "Model note: " + row["note"] + "\nDescriptive reference: " + row["ref"]
            examples.append(dict(task_id=alias, question=question, split="fit", task_type="observation",
                                 response="yes" if row["grade"] == "A" else "no", answer="yes"))
            feedback[alias] = dict(ref=row["ref"], note=row["note"], grade=row["grade"])
        folder = ROOT / ("memory_" + stage)
        folder.mkdir()
        memory = update_memory({"memories": memory}, stage, examples, folder, gen)
    save("private_memory", memory)
    prompt = ('Using the HCL memory and fit evidence, find one recurrent, visually identifiable omission. '
              'Return JSON {"when":"specific visible anatomy/modality and trigger","action":"short extra '
              'inspection/description instruction","skip":"visually identifiable exclusion","support":["id1","id2"],'
              '"contrast":"id3"}. Cite two missed findings (grade B or C) from distinct fit cases with the same '
              'pattern and a similar view where the target is absent or irrelevant. The case number precedes '
              'the dot. Do not mention diagnoses or patients in the procedural rule. Use exact evidence IDs.\n'
              + json.dumps({"abstract": [m["abstract"] for m in memory], "evidence": feedback}))
    attempts = []
    for _ in range(2):
        output = gen(prompt, 400)
        try:
            obj = json.loads(re.search(r"\{.*\}", output, re.S).group())
            for field in ("when", "action", "skip"):
                assert isinstance(obj[field], str) and 3 <= len(obj[field].strip()) <= 360
            support = obj["support"]
            assert isinstance(support, list) and len(support) == 2
            assert len({i.split(".")[0] for i in support}) == 2
            assert all(feedback[i]["grade"] in ("B", "C") for i in support)
            assert obj["contrast"] in feedback and obj["contrast"].split(".")[0] not in {i.split(".")[0] for i in support}
            add = "When " + obj["when"] + ", " + obj["action"] + ". Skip this extra instruction when " + obj["skip"] + "."
            skill, report = apply_patch_with_report(TARGET, {"edits": [{"op": "append", "content": add}]})
            assert report[0]["status"] == "applied_append" and skill != TARGET
            save("private_patch", attempts + [obj])
            return obj, skill
        except (AttributeError, AssertionError, ValueError, KeyError, TypeError) as e:
            attempts.append({"output": output, "error": type(e).__name__})
            prompt += "\nRepair your previous JSON and verify the IDs, distinct cases and grades: " + output
    save("private_patch", attempts)
    return None, None


def route(skill, img, choose):
    question = ("Inspect only this image. The procedure applies when: " + skill["when"] +
                ". Exclude when: " + skill["skip"] + ". A=applies visibly, B=absent, excluded or uncertain.")
    return choose("", {"view": img}, question, list("AB")) == "A"


def inspect(c, imgs, notes, refs, choose, blind):
    desc = "\n".join(f"View {i}: {t}" for i, t in enumerate(notes, 1) if t)
    answer = choose(context(c), imgs, "Image observations:\n" + desc +
                    "\nUsing the images and history, choose the most likely diagnosis. Reply with its single option letter.",
                    list(c["options"]))
    audits = []
    for img, note, ref in zip(imgs.values(), notes, refs):
        echo = any(re.search(r"\b" + re.escape(v.strip()) + r"\b", note, re.I) for v in c["options"].values() if v.strip())
        audits.append(dict(grade=grade(ref, note, choose), echo=echo, unsupported=blind(
            "Observation: " + note, {"view": img}, AUDIT, list("AB")) == "A"))
    return dict(answer=answer, notes=notes, audits=audits)


def compare(rows, gold, left, right):
    old = [r[left]["answer"] for r in rows.values()]
    new = [r[right]["answer"] for r in rows.values()]
    truth = [gold[k] for k in rows]
    out = paired(old, new, truth)
    delta = [int(a == g) - int(b == g) for a, b, g in zip(new, old, truth)]
    rng = random.Random(608)
    samples = sorted(sum(rng.choices(delta, k=len(delta))) * 100 / len(delta) for _ in range(3000))
    out["ci95_pp"] = [samples[75], samples[2924]]
    for group in ("all", "applicable", "outside"):
        counts = Counter()
        for r in rows.values():
            change = 0
            for eligible, a, b in zip(r["labels"], r[left]["audits"], r[right]["audits"]):
                if group != "all" and eligible != (group == "applicable"):
                    continue
                counts["views"] += 1
                for field in ("unsupported", "echo"):
                    counts[field + "_left"] += a[field]
                    counts[field + "_right"] += b[field]
                if a["grade"] == "-":
                    continue
                counts["checked"] += 1
                for name, item in (("left", a), ("right", b)):
                    counts["covered_" + name] += item["grade"] == "A"
                    counts["contradictions_" + name] += item["grade"] == "C"
                d = int(b["grade"] == "A") - int(a["grade"] == "A")
                counts["rescue"] += d > 0
                counts["loss"] += d < 0
                change += d
            counts["case_gain"] += change > 0
            counts["case_loss"] += change < 0
            counts["checked_cases"] += any(a["grade"] != "-" and (group == "all" or e == (group == "applicable"))
                                           for a, e in zip(r[left]["audits"], r["labels"]))
        out[group] = counts
    return out


def pass_gate(m):
    return (m["checked"] >= 8 and m["checked_cases"] >= 4 and m["rescue"] > m["loss"]
            and m["case_gain"] >= m["case_loss"] and all(m[k + "_right"] <= m[k + "_left"]
              for k in ("contradictions", "unsupported", "echo")))


def main():
    raw = Path(os.environ["RAW_ROOT"])
    dirs = list(map(Path, os.environ["PRIOR_RUNS"].split(os.pathsep)))
    old = json.loads((Path(os.environ["FINDINGS_RUN"]) / "private_predictions.json").read_text())
    prior, _ = build(raw, dirs)
    old2 = json.loads((Path(os.environ["SECOND_RUN"]) / "private_predictions.json").read_text())
    assert set(prior) == set(old) and len(set(old) | set(old2)) == 128
    used = set(old) | set(old2)
    choose, gen = load_vlm(with_text_generator=True)
    choose, gen = measured(choose, "main_choices"), measured(gen, "main_generations")
    os.environ["MODEL_ROOT"] = os.environ["BLIND_ROOT"]
    blind = measured(load_vlm(), "blind_choices")
    with ZipFile(raw / "images.zip") as archive:
        training, stages = {}, Counter()
        for k, c in prior.items():
            if stages[c["stage"]] >= 16:
                continue
            stages[c["stage"]] += 1
            imgs = images(c, archive)
            notes = write_notes(context(c) + "\n" + TARGET, imgs, gen)
            for j, (ref, note) in enumerate(zip(old[k]["checked"], notes)):
                alias = c["stage"] + str(stages[c["stage"]]) + "." + str(j + 1)
                training[alias] = dict(stage=c["stage"], ref=ref, note=note, grade=grade(ref, note, choose))
            if sum(stages.values()) % 8 == 0:
                print("fit", sum(stages.values()), flush=True)
        save("private_fit", training)
        obj, skill = proposals(training, gen)
        print("candidate frozen", bool(obj), flush=True)
        for phase, count in (("gate", 16), ("confirmation", 64)):
            cases, gold = build(raw, dirs, exclude=used, per_stratum=count)
            assert len(cases) == count * 2 and set(cases).isdisjoint(used)
            used.update(cases)
            labels = {k: [route(obj, img, blind) if obj else False for img in images(c, archive).values()]
                      for k, c in cases.items()}
            save("private_" + phase + "_blind", labels)
            rows = {}
            for index, (k, c) in enumerate(cases.items(), 1):
                imgs = images(c, archive)
                options = list(c["options"].values())
                refs = [check_caption(v["caption"], options, choose, gen)[0] for v in c["images"]]
                row = {"stage": c["stage"], "labels": labels[k]}
                prompts = {"targeted": context(c) + "\n" + TARGET}
                if phase == "gate":
                    prompts["generic"] = BASE
                if obj:
                    prompts["global"] = context(c) + "\n" + skill
                for arm, prompt in prompts.items():
                    row[arm] = inspect(c, imgs, write_notes(prompt, imgs, gen), refs, choose, blind)
                if phase == "confirmation":
                    routes = [route(obj, img, choose) for img in imgs.values()]
                    mixed = [row["global" if flag else "targeted"]["notes"][j] for j, flag in enumerate(routes)]
                    row["routes"] = routes
                    row["routed"] = inspect(c, imgs, mixed, refs, choose, blind)
                rows[k] = row
                if index % 8 == 0:
                    print(phase, index, flush=True)
            save("private_" + phase, rows)
            pairs = [("generic", "targeted")] if phase == "gate" else [("targeted","routed"), ("global","routed")]
            if obj:
                pairs.append(("targeted","global"))
            result = {}
            for a, b in pairs:
                result[a + "_vs_" + b] = {stage: compare({k:r for k,r in rows.items() if stage == "all" or r["stage"] == stage},
                                                          gold, a, b) for stage in ("all", "CT", "nonCT")}
            if phase == "gate":
                accepted = bool(obj) and pass_gate(result["targeted_vs_global"]["all"]["applicable"])
                result["passed"] = accepted
            else:
                confusion = Counter(("tp" if y else "fp") if pred else ("fn" if y else "tn")
                                    for r in rows.values() for y, pred in zip(r["labels"], r["routes"]))
                result["route_confusion"] = confusion
            result["calls_cumulative"] = dict(calls)
            save(phase, result)
            print(phase, "complete", flush=True)
            if phase == "gate" and not accepted:
                break


if __name__ == "__main__":
    main()
