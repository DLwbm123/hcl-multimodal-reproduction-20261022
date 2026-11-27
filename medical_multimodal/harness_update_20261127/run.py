"""Shared bounded skill editing and gated replay for two tool environments."""
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

from PIL import Image

from acquire import load_vlm

INITIAL = "Use the current evidence to answer the exact question. Call an available tool if more evidence is needed."
ARMS = ("frozen", "memory", "edited", "deployed")


def read(name):
    return json.loads(Path(name + ".json").read_text())


def save(name, value):
    dst = Path(name + ".json")
    tmp = dst.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(dst)


def parse(raw):
    text = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n\s*```", text, re.S | re.I)
    if fenced:
        text = fenced[1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def apply(response):
    if not isinstance(response, dict) or set(response) != {"op", "field", "value"}:
        return None
    text = response["value"]
    if (response["op"] != "replace" or response["field"] != "instruction"
            or not isinstance(text, str) or not 12 <= len(text.strip()) <= 420 or text.strip() == INITIAL):
        return None
    return text.strip()


def normalized(value):
    return " ".join(re.findall(r"[a-z0-9]+", str(value).lower()))


def score(kind, prediction, target):
    if prediction is None:
        return False
    if kind == "text":
        if target == "NA":
            return normalized(prediction) in ("na", "n a", "no records", "no data")
        try:
            number = float(prediction)
            return math.isfinite(number) and abs(number - float(target)) <= 0.11
        except (TypeError, ValueError):
            return False
    if normalized(prediction) == normalized(target):
        return True
    slots = {",", ";"}
    return (any(ch in str(target) for ch in slots) and
            set(map(normalized, re.split(r"[;,]", str(prediction)))) ==
            set(map(normalized, re.split(r"[;,]", str(target)))))


def serialize(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def recorded_context(case, mode, image, generate):
    if mode == "visual":
        answer = generate(
            "Inspect the current endoscopy image and give a short visual observation relevant to this question. "
            "Describe only visible evidence. Question: " + case["question"], 56, image
        )
        return dict(observation=answer.strip())
    return dict(rows=case["rows"][:4], has_more=len(case["rows"]) > 4,
                columns=["hours_before_now", "glucose_mg_dl"], sort="newest first")


def episode(case, mode, image, first, guidance, examples, generate):
    q = case["question"]
    observations = [{"tool": "initial", "receipt": first}]
    attempted = []
    answer = None
    if mode == "visual":
        contract = ('Output one JSON object: {"answer":"short answer"} or {"tool":"verify"} to get '
                    'one more task-focused observation of the current image. Answer the requested visual fact only.')
    else:
        contract = ('Output one JSON object: {"tool":"more"} to read the remaining glucose rows; '
                    '{"tool":"mean"} to finish with the mean of currently loaded rows in the last 24 hours; '
                    '{"tool":"latest"} to finish with their latest value; or {"answer":"NA"} if the requested value '
                    'is missing. Direct numerical {"answer":"123.4"} is permitted. Pages are newest first.')
    base = f"Question: {q}\nWorkflow skill: {guidance}\n{contract}"
    if examples:
        base += "\nHistorical feedback for other cases. Never copy their answer for this case. " + serialize(examples)
    for _ in range(2):
        raw = generate(base + "\nCurrent receipts: " + serialize(observations), 72, image)
        obj = parse(raw)
        if not isinstance(obj, dict):
            attempted.append({"invalid": raw[:160]})
            break
        attempted.append(obj)
        if set(obj) == {"answer"} and type(obj["answer"]) in (str, float, int):
            answer = obj["answer"]
            break
        if set(obj) != {"tool"} or type(obj.get("tool")) is not str:
            break
        tool = obj["tool"]
        if mode == "visual" and tool == "verify" and len(observations) == 1:
            text = generate(
                "Look again at the current entire image. Verify the visual fact the question asks for, checking "
                "count, presence, or instrument identity as applicable. Report only visible evidence. " + q, 56, image
            )
            observations.append({"tool": "verify", "receipt": text.strip()})
        elif mode == "text" and tool == "more" and len(observations) == 1:
            observations.append({"tool": "more", "receipt": {"rows": case["rows"][4:], "has_more": False}})
        elif mode == "text" and tool in ("mean", "latest"):
            rows = case["rows"] if len(observations) == 2 else first["rows"]
            if tool == "mean":
                result = [v for age, v in rows if age <= 24]
                answer = sum(result) / len(result) if result else "NA"
            else:
                answer = rows[0][1] if rows else "NA"
            observations.append({"tool": tool, "receipt": answer})
            break
        else:
            observations.append({"tool": tool, "receipt": "invalid, unavailable, or repeated action"})
            break
    return dict(answer=answer, actions=attempted, observations=observations)


def history_item(case, trace, label, mode):
    def short(receipt):
        return receipt[:180] if isinstance(receipt, str) else receipt
    return dict(question=case["question"], initial=short(trace["observations"][0]["receipt"]),
                final_observation=short(trace["observations"][-1]["receipt"]),
                actions=trace["actions"], attempt=trace["answer"], verified=label,
                passed=score(mode, trace["answer"], label))


def words(text):
    return set(re.findall(r"[a-z]+", text.lower()))


def retrieve(memory, case):
    q = words(case["question"])
    return sorted(memory, key=lambda r: (
        -len(q & words(r["question"])) / max(1, len(q | words(r["question"]))), r["passed"]
    ))[:2]


def behavior(trace):
    return trace["actions"], trace["answer"]


def pairs(rows, mode, ref, candidate):
    total = Counter(n=len(rows))
    for row in rows:
        a, b = row["arms"][ref], row["arms"][candidate]
        x, y = score(mode, a["answer"], row["label"]), score(mode, b["answer"], row["label"])
        total.update(before=int(x), after=int(y), gain=int(y and not x), loss=int(x and not y),
                     retained=int(x and y), behavior_changed=int(behavior(a) != behavior(b)),
                     tools_before=len(a["observations"]), tools_after=len(b["observations"]))
    return total


def main():
    import torch

    mode = os.environ["MODE"]
    assert mode in ("visual", "text")
    cases = read("cases")
    labels = read("labels")
    _, generate = load_vlm(with_text_generator=True)
    progress = []
    memory = []
    for case in (r for r in cases if r["role"] == "fit"):
        if mode == "visual":
            with Image.open(case["image"]) as img:
                image = img.convert("RGB")
        else:
            image = None
        first = recorded_context(case, mode, image, generate)
        baseline = episode(case, mode, image, first, INITIAL, [], generate)
        memory.append(history_item(case, baseline, labels[case["id"]], mode))
        print("fit", len(memory), flush=True)
    training = sorted(memory, key=lambda x: x["passed"])
    errors = [x for x in training if not x["passed"]][:4]
    successes = [x for x in training if x["passed"]][:2]
    candidates = []
    if errors:
        for perspective in ("Identify what the failed decisions missed.", "Identify an overly broad or premature decision."):
            prompt = ("Revise one existing workflow skill instruction from TRAINING feedback. " + perspective
                      + " Preserve successful behavior. State a general decision rule grounded in the current case; "
                        "never copy historical answers or rely on labels at runtime. Maximum 420 characters. "
                        'Return only {"op":"replace","field":"instruction","value":"..."}.'
                      + "\nCurrent skill: " + INITIAL + "\nTraining feedback: " + serialize(errors + successes))
            for attempt in range(2):
                raw = generate(prompt, 160)
                patch = apply(parse(raw))
                if patch is not None:
                    if patch not in candidates:
                        candidates.append(patch)
                    break
                prompt += "\nPrevious output did not match the format. Return the exact JSON object only."
    save("versions", dict(initial=INITIAL, candidates=candidates))
    print("valid_candidates", len(candidates), flush=True)

    def evaluate(case, edited):
        if mode == "visual":
            with Image.open(case["image"]) as img:
                image = img.convert("RGB")
        else:
            image = None
        first = recorded_context(case, mode, image, generate)
        historical = retrieve(memory, case)
        arms = {"frozen": episode(case, mode, image, first, INITIAL, [], generate),
                "memory": episode(case, mode, image, first, INITIAL, historical, generate)}
        arms.update({f"edit{i}": episode(case, mode, image, first, skill, historical, generate)
                     for i, skill in edited})
        item = dict(id=case["id"], role=case["role"], label=labels[case["id"]], arms=arms)
        progress.append(item)
        save("private_progress", progress)
        print("evaluate", case["role"], len(progress), flush=True)
        return item

    checked = [evaluate(c, list(enumerate(candidates))) for c in cases if c["role"] in ("gate", "anchor")]
    gate_rows = [r for r in checked if r["role"] == "gate"]
    anchor_rows = [r for r in checked if r["role"] == "anchor"]
    gate = []
    for i in range(len(candidates)):
        key = f"edit{i}"
        current = pairs(gate_rows, mode, "memory", key)
        old = pairs(anchor_rows, mode, "memory", key)
        original = pairs(gate_rows, mode, "frozen", key)
        old_frozen = pairs(anchor_rows, mode, "frozen", key)
        eligible = current["after"] > max(current["before"], original["before"]) and not (
            old["loss"] or old_frozen["loss"]
        )
        gate.append(dict(candidate=i, current=current, anchors=old, anchor_vs_frozen=old_frozen, accepted=eligible))
    selection = max((r for r in gate if r["accepted"]),
                    key=lambda r: (r["current"]["after"], -r["candidate"]), default=None)
    shadow = selection or max(gate, key=lambda r: (r["current"]["after"], -r["candidate"]), default=None)
    selected = selection["candidate"] if selection is not None else None
    tested = shadow["candidate"] if shadow is not None else None
    memory_gate = pairs(gate_rows, mode, "frozen", "memory")
    memory_anchor = pairs(anchor_rows, mode, "frozen", "memory")
    fallback = "memory" if memory_gate["after"] >= memory_gate["before"] and not memory_anchor["loss"] else "frozen"
    save("commit", dict(selected=selected, shadow=tested, gate=gate, fallback=fallback))
    test = [evaluate(c, [] if tested is None else [(tested, candidates[tested])])
            for c in cases if c["role"] == "test"]
    for row in test:
        arm = row["arms"]
        arm["edited"] = arm[f"edit{tested}"] if tested is not None else arm["memory"]
        arm["deployed"] = arm[f"edit{selected}"] if selected is not None else arm[fallback]
    save("result", dict(mode=mode, splits=read("split_summary"), training=Counter(
        total=len(memory), correct=sum(r["passed"] for r in memory), failed=sum(not r["passed"] for r in memory)),
        valid_patches=len(candidates), selected=selected, shadow=tested, gate=gate, fallback=fallback,
        memory_gate=memory_gate, memory_anchor=memory_anchor,
        test={arm: pairs(test, mode, "frozen", arm) for arm in ARMS},
        edit_vs_memory=pairs(test, mode, "memory", "edited"),
        cuda_allocated_mib=round(torch.cuda.max_memory_allocated()/1048576, 1),
        cuda_reserved_mib=round(torch.cuda.max_memory_reserved()/1048576, 1)))


if __name__ == "__main__":
    assert apply({"op":"replace","field":"instruction","value":"Use current tool evidence."}) is not None
    assert apply({"op":"replace","field":"tool","value":"unsupported"}) is None
    main()
