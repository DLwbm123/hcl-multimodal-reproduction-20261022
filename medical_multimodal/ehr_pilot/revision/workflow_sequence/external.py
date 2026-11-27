"""Frozen independent-source evaluation on VitalDB patients."""

import csv
import gzip
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path
from statistics import mean

import workflow as w
from nearest_value import execute as numeric_retrieval

b, p, d, v, old = w.b, w.p, w.d, w.v, w.old
base_execute = w.execute


def data_rows(filename):
    path = Path(filename)
    with path.open("rb") as source:
        compressed = source.read(2) == b"\x1f\x8b"
    opener = gzip.open if compressed else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as source:
        yield from csv.DictReader(source)


def freeze():
    by_subject = {}
    for row in data_rows("cases.csv.gz"):
        if not row["caseid"] or not row["subjectid"] or not row["caseend"]:
            continue
        case, subject, end = int(row["caseid"]), int(row["subjectid"]), float(row["caseend"])
        if end <= 0 or not math.isfinite(end):
            continue
        initial = by_subject.get(subject)
        if not initial or case < initial[0]:
            by_subject[subject] = (case, end)
    anchors = {case: end for case, end in by_subject.values()}
    measurements = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    for row in data_rows("labs.csv.gz"):
        name = row["name"]
        if name not in ("k", "gluc") or not row["caseid"] or not row["dt"] or not row["result"]:
            continue
        case = int(row["caseid"])
        if case not in anchors:
            continue
        try:
            when, val = float(row["dt"]), float(row["result"])
        except ValueError:
            continue
        if math.isfinite(when) and math.isfinite(val) and when <= anchors[case]:
            measurements[case][name][when].add(val)

    pools = defaultdict(list)
    ambiguous = 0
    for case in sorted(anchors):
        k = measurements[case]["k"]
        if k and len(k[max(k)]) != 1:
            ambiguous += 1
            continue
        branch = "missing" if not k else "write" if next(iter(k[max(k)])) < 3.5 else "no_write"
        pools[branch].append(case)
    generator = random.Random(26112721)
    chosen = []
    for branch in ("write", "no_write", "missing"):
        pool = pools[branch]
        generator.shuffle(pool)
        assert len(pool) >= 40
        chosen.extend(pool[:40])
    generator.shuffle(chosen)
    assert len(set(chosen)) == 120
    templates = {kind: next(c for c in v.ROWS.values() if d.kind(c) == kind) for kind in (6, 9)}
    subjects = []
    for i, source in enumerate(chosen):
        patient = f"S88771{i:04d}"
        requests = []
        for kind, name, unit in ((9, "k", "mmol/L"), (6, "gluc", "mg/dL")):
            case = dict(templates[kind], id=f"task{kind}_{i}", sol=None, eval_MRN=patient,
                        instruction=re.sub(r"\bS\d+\b", patient, templates[kind]["instruction"]))
            rows = [dict(resourceType="Observation", subject={"reference": "Patient/" + patient},
                         code={"coding": [{"code": d.CODES[kind]}]}, valueQuantity={"value": next(iter(values)), "unit": unit},
                         effectiveDateTime=(d.MONITOR+timedelta(seconds=stamp-anchors[source])).isoformat())
                    for stamp, values in sorted(measurements[source][name].items()) if len(values) == 1]
            requests.append((case, w.observation(case, rows)))
        subjects.append(requests)
    p.save("external_cases.json", subjects)
    p.save("external_counts.json", dict(n_subjects=120, source_subjects=len(by_subject),
           eligible={key: len(pool) for key, pool in pools.items()}, ambiguous_excluded=ambiguous))


def action_for(case, msgs):
    text = p.generate([msgs], limit=80)[0].strip().strip("`")
    raw = text[4:].strip() if text.startswith("GET ") else ""
    url = p.BASE.rstrip("/") + raw.removeprefix("/fhir") if raw.startswith("/fhir/") else raw
    return old.same_query(url, d.query(case), ("_format", "_count", "_sort")), text


def infer():
    v.setup_model()
    partition = int(os.environ["PART"])
    subjects = old.read("external_cases")[partition::2]
    name = f"x{partition}"
    records = old.read(name) if Path(name + ".json").exists() else []
    cases = [case for subject in subjects for case, _ in subject]
    assert [x["id"] for x in records] == [x["id"] for x in cases[:len(records)]]
    instruction = (
        "Generate the first GET request for this question. The read-only tool is "
        "GET /fhir/Observation?patient=<identifier>&code=<abbreviation>. "
        "Laboratory abbreviations are MG for magnesium, GLU for glucose, K for potassium, A1C for hemoglobin A1c. "
        "Use precisely the current patient identifier and lab. Output only the GET line.\n"
    )
    for case in cases[len(records):]:
        msgs = [dict(role="user", content=instruction + case["instruction"])]
        for attempt in range(1, 3):
            passed, text = action_for(case, msgs)
            if passed:
                break
            msgs.extend([dict(role="assistant", content=text),
                         dict(role="user", content="Invalid or wrong query. Try again with the documented GET syntax.")])
        records.append(dict(id=case["id"], accepted=passed, attempts=attempt))
        p.save(name + ".json", records)
        if len(records) % 8 == 0 or len(records) == len(cases):
            print(partition, len(records), "/", len(cases), flush=True)


def policy(model, case, obs, method, conditioned=True, overrides=None):
    runner = numeric_retrieval if d.kind(case) == 9 and method == "retrieval" else base_execute
    return runner(model, case, obs, method, conditioned, overrides)


def interval(clusters):
    rng = random.Random(26112722)
    draws = sorted(mean(x for c in rng.choices(clusters, k=len(clusters)) for x in c)
                   for i in range(4000))
    return [draws[100], draws[3899]]


def report():
    from unittest.mock import patch
    p.ref = p.scorer()
    model = w.fit(w.bank_rows()[0])
    people = old.read("external_cases")
    qualified = {x["id"]: x for n in range(2) for x in old.read(f"x{n}")}
    assert {c["id"] for person in people for c, _ in person} == qualified.keys()
    totals = defaultdict(Counter)
    deltas = defaultdict(lambda: defaultdict(list))
    for person in people:
        for case, receipt in person:
            key = case["id"]
            plans = {tag: policy(model, case, receipt, tag) for tag in ("retrieval", "induction")}
            plan, truth = b.expected(case, receipt)
            assert b.score(case, receipt, truth, truth)["full"]
            for mode in ("given", "query"):
                ok = mode == "given" or qualified[key]["accepted"]
                scores = {tag: b.score(case, receipt, value[1] if ok else None, truth)
                          for tag, value in plans.items()}
                a = scores["retrieval"]["full"]
                z = scores["induction"]["full"]
                for group in ("all", p.family(case), "write" if truth[0] else "no_write"):
                    stat = totals[f"{mode}_{group}"]
                    stat.update(n=1, observed=int(ok), plan=int(ok and plans["induction"][0] == plan),
                                retrieval=int(a), induction=int(z), gain=int(z and not a), loss=int(a and not z),
                                retained=int(a and z), retrieval_wrong_write=int(scores["retrieval"]["wrong_write"]),
                                induction_wrong_write=int(scores["induction"]["wrong_write"]))
                    deltas[mode, group][case["eval_MRN"]].append(int(z) - int(a))
    result = dict(data=old.read("external_counts"), totals=totals,
                  clusters={f"{mode}_{group}": dict(n=len(values), ci_95=interval(list(values.values())))
                            for (mode, group), values in deltas.items()},
                  query_attempts=sum(x["attempts"] for x in qualified.values()),
                  first_query_correct=sum(x["attempts"] == 1 and x["accepted"] for x in qualified.values()))
    k = [person[0] for person in people]
    with patch.object(w, "execute", policy):
        result["composition_given"] = w.compose(k, model)
        result["composition_query"] = w.compose([(c, o) for c, o in k if qualified[c["id"]]["accepted"]], model)
    p.save("external_result.json", result)


if __name__ == "__main__":
    {"freeze": freeze, "infer": infer, "report": report}[sys.argv[1]]()
