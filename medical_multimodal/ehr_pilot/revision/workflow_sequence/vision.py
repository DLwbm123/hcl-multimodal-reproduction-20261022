"""Paired transfer of a conditional tool skill to binary imaging questions."""

import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from PIL import Image

from acquire import load_vlm
from grounding import boxes, locate
from observation import observe
from signatures import signature

OPTIONS = ("stop", "fixed", "targeted")


def read(path):
    return json.loads(Path(path).read_text())


def save(name, value):
    temp = Path(name + ".tmp")
    temp.write_text(json.dumps(value, indent=1))
    temp.replace(name + ".json")


def binary(answer):
    text = str(answer).strip().lower().rstrip(".")
    return {"yes": "A", "no": "B"}.get(text)


def freeze():
    historical = Path("history")
    previous = read(historical / "inputs.json")
    labels = read(historical / "labels.json")
    eligible = [dict(r, answer=binary(labels[r["id"]])) for r in previous if binary(labels[r["id"]])]
    random.Random(26112718).shuffle(eligible)
    retained = eligible[:90]
    split = 2 * len(retained) // 3
    for i, item in enumerate(retained):
        item["stage"] = "train" if i < split else "gate"

    root = Path("source")
    all_rows = read(root / "VQA_RAD Dataset Public.json")
    # Union of study groups linked by the same source image, including excluded non-binary questions.
    parents = {}
    def group(node):
        parents.setdefault(node, node)
        while node != parents[node]:
            node = parents[node]
        return node

    eligible_by_case = []
    for row in all_rows:
        name = row["image_name"].strip()
        case = parse_qs(urlparse(row["image_case_url"].strip()).query).get("id", [""])[0].lower()
        if not name or not case:
            continue
        parents[group("img:" + name)] = group("case:" + case)
        if row["phrase_type"] in ("freeform", "test_freeform") and binary(row["answer"]) and row["question"].strip():
            eligible_by_case.append((name, row))
    excluded = {group("img:" + Path(x["path"]).name) for x in previous}
    seen = [signature(Path(r["path"])) for r in previous]
    old_images = {image for line in Path("old_data/tasks.jsonl").read_text().splitlines()
                  for image in json.loads(line)["images"]}
    seen.extend(signature(Path("old_data") / path) for path in sorted(old_images))
    grouped = defaultdict(list)
    for name, row in eligible_by_case:
        key = group("img:" + name)
        if key not in excluded:
            grouped[key].append((name, row))
    rng = random.Random(26112719)
    keys = sorted(grouped)
    rng.shuffle(keys)
    fresh = []
    for key in keys:
        name, row = rng.choice(sorted(grouped[key], key=lambda pair: str(pair[1]["qid"])))
        path = root / "images" / name
        sig = signature(path)
        if any((x ^ sig).bit_count() <= 10 for x in seen):
            continue
        seen.append(sig)
        fresh.append(dict(id=f"x{len(fresh)}", path=str(path), q=row["question"].strip(),
                          answer=binary(row["answer"]), stage="test"))
    assert fresh and len(retained) >= 60
    items = retained + fresh
    save("cases", [{key: value for key, value in item.items() if key != "answer"} for item in items])
    save("labels", {item["id"]: item["answer"] for item in items})
    save("composition", dict(groups=Counter(x["stage"] for x in items), excluded_groups=len(excluded),
                             candidate_groups=len(keys), prior_exact_paths=len(previous)))


def infer():
    part = int(os.environ["PART"])
    cases = read("cases.json")[part::2]
    choose, gen = load_vlm(with_text_generator=True)
    results = read("predictions.json") if Path("predictions.json").exists() else []
    assert [r["id"] for r in results] == [r["id"] for r in cases[:len(results)]]
    instruction = "Answer the clinical question using the current evidence. A means yes. B means no. Select one."
    for item in cases[len(results):]:
        path = Path(item["path"])
        with Image.open(path if path.is_absolute() else Path("..") / path) as img:
            image = img.convert("RGB")
        question = item["q"]
        global_evidence = observe(gen, question, image)
        prompt = f"Clinical question: {question}\nInitial tool evidence: {global_evidence}"
        stop = choose(prompt, {}, instruction, ["A", "B"])
        center = boxes(image.size)[-1]
        target = locate(choose, prompt, image, set())
        assert target is not None
        evidence = {center: observe(gen, question, image, center)}
        if target not in evidence:
            evidence[target] = observe(gen, question, image, target)
        out = {}
        for box, value in evidence.items():
            out[box] = choose(prompt + "\nCurrent localized evidence: " + value, {}, instruction, ["A", "B"])
        results.append(dict(id=item["id"], initial=global_evidence, stop=stop, fixed=out[center],
                            targeted=out[target], fixed_nonempty=bool(evidence[center]),
                            targeted_nonempty=bool(evidence[target]), same_box=target == center,
                            targeted_changed=out[target] != stop, fixed_changed=out[center] != stop))
        save("predictions", results)
        if len(results) % 8 == 0 or len(results) == len(cases):
            print("images", part, len(results), "/", len(cases), flush=True)


def features(question, evidence):
    return (
        bool(re.search(r"\b(left|right|upper|lower|apex|base|side|lobe|location|where)\b", question, re.I)),
        bool(re.search(r"\b(uncertain|unclear|limited|cannot|not visible|difficult|indeterminate|not enough)\b",
                       evidence, re.I)) or not evidence,
    )


def words(text):
    return set(re.findall(r"[a-z]+", text.lower()))


def compare():
    cases = read("cases.json")
    predictions = {r["id"]: r for i in range(2) for r in read(f"s{i}/predictions.json")}
    labels = read("labels.json")
    assert set(predictions) == set(labels) == {r["id"] for r in cases}
    stages = defaultdict(list)
    for item in cases:
        key = item["id"]
        row = predictions[key]
        stages[item["stage"]].append((item, row, labels[key]))
    prior = stages["train"]

    def best(examples):
        return max(OPTIONS, key=lambda action: sum(r[action] == answer for _, r, answer in examples))

    global_action = best(prior)
    train_groups = defaultdict(list)
    for record in prior:
        item, row, _ = record
        train_groups[features(item["q"], row["initial"])].append(record)
    policies = {key: best(group) if len(group) >= 6 else global_action for key, group in train_groups.items()}

    def decisions(item, row):
        cue = words(item["q"] + " " + row["initial"])
        def similarity(record):
            query, observation, _ = record
            target = words(query["q"] + " " + observation["initial"])
            return len(cue & target) / max(1, len(cue | target))
        return dict(stop="stop", fixed="fixed", targeted="targeted",
                    retrieval=best([max(prior, key=similarity)]),
                    skill=policies.get(features(item["q"], row["initial"]), global_action))

    result = dict(data=read("composition.json"), train_default=global_action,
                  rules={str(key): dict(n=len(train_groups[key]), action=action) for key, action in policies.items()})
    for stage in ("gate", "test"):
        tallies = defaultdict(Counter)
        for record in stages[stage]:
            item, row, target = record
            actions = decisions(item, row)
            outcomes = {method: row[action] == target for method, action in actions.items()}
            for method, action in actions.items():
                x = outcomes[method]
                tallies[method].update(n=1, correct=x, extra_calls=action != "stop",
                                       model_calls=2 + {"stop": 0, "fixed": 2, "targeted": 3}[action],
                                       changed=row[action] != row["stop"],
                                       nonempty=bool(row["initial"]) if action == "stop" else row[action + "_nonempty"])
                for ref in ("stop", "fixed", "retrieval"):
                    y = outcomes[ref]
                    tallies[method].update({f"gain_vs_{ref}": x and not y, f"loss_vs_{ref}": y and not x,
                                            f"retained_vs_{ref}": x and y})
        result[stage] = tallies
    result["gate_accept"] = (
        result["gate"]["skill"]["correct"] > result["gate"]["stop"]["correct"]
        and result["gate"]["skill"]["correct"] >= result["gate"]["fixed"]["correct"]
        and result["gate"]["skill"]["correct"] >= result["gate"]["retrieval"]["correct"]
    )
    result["paired_crops"] = dict(n=len(stages["test"]),
                                   same_box=sum(r["same_box"] for _, r, _ in stages["test"]),
                                   fixed_nonempty=sum(r["fixed_nonempty"] for _, r, _ in stages["test"]),
                                   targeted_nonempty=sum(r["targeted_nonempty"] for _, r, _ in stages["test"]))
    save("image_result", result)


if __name__ == "__main__":
    {"freeze": freeze, "infer": infer, "report": compare}[sys.argv[1]]()
