"""Frozen routing, native queries, workflow composition and sequential updates."""

import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import binding as b

p, d, v, old = b.p, b.d, b.v, b.prior
A = {4, 7}
B = {5, 6, 9, 10}


def bank_rows():
    panels = b.datasets()
    history = panels["medagent"]["train"]
    potassium = panels["eicu"]["train"]
    magexamples = [c for c in old.read("magnesium/cases") if c["stage"] == "build"]
    resources = defaultdict(list)
    for item in old.read("magnesium/resources"):
        if item["resourceType"] == "Observation":
            resources[item["subject"]["reference"]].append(item)
    # Fixed 16 per historical stratum, following the recorded manifest order.
    counts = Counter()
    magnesium = []
    for case in magexamples:
        counts[case["group"]] += 1
        if counts[case["group"]] <= 16:
            magnesium.append((case, observation(case, resources["Patient/" + case["eval_MRN"]])))
    assert len(magnesium) == 48
    return history + potassium + magnesium, panels


def observation(case, rows):
    times = sorted((datetime.fromisoformat(r["effectiveDateTime"]), r["valueQuantity"]["value"]) for r in rows)
    stamp, value = times[-1] if times else (None, 0)
    hours = (d.MONITOR - stamp).total_seconds() / 3600 if stamp else 0
    return dict(url=d.query(case), rows=rows,
                features=[len(rows), sum(0 <= (d.MONITOR-t).total_seconds() <= 86400 for t, _ in times),
                          hours, value])


def select_plan(model, kind, obs, conditioned):
    rules = model[2]
    if kind not in rules:
        return None
    full, branches = rules[kind]
    rule = branches.get(not obs["rows"], full) if conditioned else full
    return old.predict(rule, obs["features"])


def fit(examples):
    groups, compiled, single = b.train(examples)
    conditioned = defaultdict(lambda: defaultdict(list))
    for (kind, plan), group in groups.items():
        for features, view, payload in group:
            conditioned[kind][features[0] == 0].append((features, plan, view["patient"]))
    rules = {kind: (rule, {key: old.train(rows) for key, rows in conditioned[kind].items()})
             for kind, rule in single.items()}
    return groups, compiled, rules


def execute(model, case, obs, method, conditioned=True, overrides=None):
    kind = d.kind(case)
    plan = select_plan(model, kind, obs, conditioned)
    if plan is None:
        return None, None
    group = model[0][kind, plan]
    # Frozen nearest demonstrator, using the same normalized feature distance as the previous run.
    scales = [max(x) - min(x) or 1 for x in zip(*(entry[0] for entry in group))]
    _, view, observed = min(
        group, key=lambda entry: sum(abs(x-y) / s for x, y, s in zip(entry[0], obs["features"], scales))
    )
    program = model[1].get((kind, plan)) if method == "induction" else None
    if overrides and plan in overrides:
        program = overrides[plan]
    try:
        return plan, b.render(program or b.compile_field([observed], [view]), b.fields(case, obs))
    except ValueError:
        return plan, None


def fresh_cases():
    partition = old.read("archive/partition")
    exposed = {v.ROWS[k]["eval_MRN"] for seq in partition.values() for k in seq}
    selected = [c for c in v.ROWS.values()
                if c.get("eval_MRN") and c["eval_MRN"] not in exposed and p.family(c) in
                {f"task{i}" for i in A | B}]
    assert len(selected) == 27
    return selected


def native_obs(case, url):
    if not old.same_query(url, d.query(case), ("_format", "_count", "_sort")):
        return None
    # Run the whole, validated patient/lab query against the read-only loopback service.
    # Optional presentation flags do not change the full retrieval in this adapter.
    reply, read = d.model_fetch(d.query(case))
    if read is None:
        return None
    return old.observation(case, {"history": [["agent", "GET " + d.query(case)], ["user", reply]]})


def forecast_native():
    v.setup_model()
    index = int(os.environ["PART"])
    records = []
    cases = fresh_cases()[index::2]
    instructions = (
        "Generate the first GET request for this question. The read-only tool is "
        "GET /fhir/Observation?patient=<identifier>&code=<abbreviation>. "
        "Laboratory abbreviations are MG for magnesium, GLU for glucose, K for potassium, A1C for hemoglobin A1c. "
        "Use precisely the current patient identifier and lab. Output only the GET line.\n"
    )
    for case in cases:
        msg = [dict(role="user", content=instructions + case["instruction"])]
        obs = None
        for attempt in range(1, 3):
            action = p.generate([msg], limit=80)[0].strip().strip("`")
            if action.startswith("GET "):
                raw = action[4:].strip()
                url = p.BASE.rstrip("/") + raw.removeprefix("/fhir") if raw.startswith("/fhir/") else raw
                obs = native_obs(case, url)
            if obs is not None:
                break
            msg.extend([dict(role="assistant", content=action),
                        dict(role="user", content="Invalid or wrong query. Try again with the documented GET syntax.")])
        records.append(dict(id=case["id"], observation=obs, attempts=attempt))
        p.save(f"query_{index}.json", records)
        print("query", index, len(records), "/", len(cases), flush=True)


def summaries(cases, model, source):
    totals = defaultdict(Counter)
    rows = {}
    for case, obs in cases:
        reference_obs = obs if obs is not None else native_obs(case, d.query(case))
        assert reference_obs is not None
        # Process both candidate policies before computing reference payloads.
        guesses = {name: execute(model, case, obs, method, conditioned) if obs is not None else (None, None)
                   for name, method, conditioned in [
                       ("stump_retrieval", "retrieval", False), ("stump_induction", "induction", False),
                       ("conditioned_retrieval", "retrieval", True), ("conditioned_induction", "induction", True)
                   ]}
        target, truth = b.expected(case, reference_obs)
        assert b.score(case, reference_obs, truth, truth)["full"]
        result = {name: b.score(case, reference_obs, guess, truth)
                  for name, (plan, guess) in guesses.items()}
        rows[case["id"]] = {name: s["full"] for name, s in result.items()}
        for key in ("all", p.family(case), "write" if truth[0] else "no_write"):
            stat = totals[key]
            stat.update(n=1, observed=obs is not None)
            for name, (plan, guess) in guesses.items():
                stat.update({f"{name}_plan": plan == target})
                stat.update({f"{name}_{metric}": value for metric, value in result[name].items()})
            a, z = (result[f"conditioned_{name}"]["full"] for name in ("retrieval", "induction"))
            stat.update(gain=z and not a, loss=a and not z, retained=a and z)
    return dict(source=source, totals=totals)


def compose(cases, model):
    """A test-only conjunction of potassium action and mean of that same current series."""
    totals = defaultdict(Counter)
    for case, obs in cases:
        numeric_case = dict(case, id="task6_compose")
        first_view = obs
        forecasts = {}
        for method in ("retrieval", "induction"):
            plan, w = execute(model, case, first_view, method)
            nplan, n = execute(model, numeric_case, first_view, method)
            forecasts[method] = (w, n)
        _, truth = b.expected(case, obs)
        moments = [(datetime.fromisoformat(x["effectiveDateTime"]), x["valueQuantity"]["value"]) for x in obs["rows"]]
        values = [val for stamp, val in moments if 0 <= (d.MONITOR-stamp).total_seconds() <= 86400]
        target = sum(values) / len(values) if values else -1
        flags = {}
        nums = {}
        for method, (w, n) in forecasts.items():
            write_ok = w is not None and b.score(case, obs, w, truth)["full"]
            nums[method] = bool(n and not n[0] and len(n[1]) == 1 and b.numeric(n[1][0])
                                and (n[1][0] == target if target == -1 else abs(n[1][0]-target) < 0.1))
            flags[method] = write_ok and nums[method]
        for key in ("all", "write" if truth[0] else "no_write"):
            totals[key].update(n=1, retrieval=flags["retrieval"], induction=flags["induction"],
                               gain=flags["induction"] and not flags["retrieval"],
                               loss=flags["retrieval"] and not flags["induction"],
                               retained=all(flags.values()), numeric_retrieval=nums["retrieval"],
                               numeric_induction=nums["induction"])
    return totals


def evaluate_skill(cases, model, methods, overrides=None):
    flags = []
    for case, obs in cases:
        payloads = [execute(model, case, obs, method, overrides=overrides)[1] for method in methods]
        _, truth = b.expected(case, obs)
        flags.extend([b.score(case, obs, payload, truth)["full"] for payload in payloads])
    return flags


def sequential(examples, anchor, fresh):
    old_bank = fit([(c, o) for c, o in examples if d.kind(c) in A])
    new = fit([(c, o) for c, o in examples if d.kind(c) in B])
    both = fit(examples)
    # Candidate consolidates identical output signatures across tasks, using the new stage's bindings.
    # Its scoped variant retains the previous skill programs on prior task IDs.
    shared = {plan: program for (kind, plan), program in sorted(new[1].items()) if kind == 6}
    a_anchor = [(c, o) for c, o in anchor if d.kind(c) in A]
    b_anchor = [(c, o) for c, o in anchor if d.kind(c) in B]
    old_pass = evaluate_skill(a_anchor, old_bank, ["induction"])
    candidates = {}
    for key, scope in (("shared", shared), ("scoped", None)):
        before = evaluate_skill(a_anchor, both, ["induction"], scope)
        after = evaluate_skill(b_anchor, both, ["induction"], scope)
        candidates[key] = dict(old=sum(before), new=sum(after),
                               lost=sum(x and not y for x, y in zip(old_pass, before)))
    ungated = max(candidates, key=lambda key: candidates[key]["new"])
    eligible = [key for key in candidates if candidates[key]["lost"] == 0]
    guarded = max(eligible, key=lambda key: candidates[key]["new"]) if eligible else "rollback"
    a = [(c, o) for c, o in fresh if d.kind(c) in A]
    z = [(c, o) for c, o in fresh if d.kind(c) in B]
    baseline = evaluate_skill(a, old_bank, ["retrieval"])
    stage_a = evaluate_skill(a, old_bank, ["induction"])
    result = dict(anchor=candidates, selected=dict(ungated=ungated, guarded=guarded),
                  a_n=len(a), b_n=len(z), a_retrieval=sum(baseline), a_after_a=sum(stage_a),
                  a_acquired=sum(y and not x for x, y in zip(baseline, stage_a)),
                  b_before=sum(evaluate_skill(z, old_bank, ["induction"])))
    for name, choice in (("ungated", ungated), ("guarded", guarded)):
        bank = old_bank if choice == "rollback" else both
        scope = shared if choice == "shared" else None
        aa = evaluate_skill(a, bank, ["induction"], scope)
        bb = evaluate_skill(z, bank, ["induction"], scope)
        result[name] = dict(a=sum(aa), b=sum(bb),
                            old_lost=sum(x and not y for x, y in zip(stage_a, aa)),
                            acquired_retained=sum(not x and y and z for x, y, z in zip(baseline, stage_a, aa)))
    return result


def report():
    p.ref = p.scorer()
    examples, panels = bank_rows()
    model = fit(examples)
    fresh = [(c, native_obs(c, d.query(c))) for c in fresh_cases()]
    assert all(o is not None for c, o in fresh)
    raw = {row["id"]: row for i in range(2) for row in old.read(f"query_{i}")}
    assert set(raw) == {c["id"] for c, _ in fresh}
    query = [(c, raw[c["id"]]["observation"]) for c, _ in fresh]
    anchors = panels["medagent"]["valid"]
    eicu = panels["eicu"]["valid"]
    output = dict(
        reference_controls=len(fresh),
        native_subjects=len({c["eval_MRN"] for c, _ in fresh}),
        native_query_attempts=sum(r["attempts"] for r in raw.values()),
        native_query_first_pass=sum(r["observation"] is not None and r["attempts"] == 1 for r in raw.values()),
        dev_medagent=summaries(anchors, model, "reused development"),
        dev_eicu=summaries(eicu, model, "reused development"),
        native_given=summaries(fresh, model, "patient-disjoint benchmark receipt replay"),
        native_end_to_end=summaries(query, model, "patient-disjoint question-to-loopback replay"),
        unseen_composition=compose(eicu, model),
        continual=sequential(examples, anchors, fresh),
    )
    p.save("ehr_result.json", output)


if __name__ == "__main__":
    {"infer": forecast_native, "report": report}[sys.argv[1]]()
