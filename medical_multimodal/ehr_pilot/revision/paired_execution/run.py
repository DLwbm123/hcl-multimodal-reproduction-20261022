"""Paired binding evaluation over cached, verified observations."""

import json
import os
import re
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import m as prior

p, d, v = prior.p, prior.d, prior.v
NO_BINDING = None


def fields(case, obs):
    clock = re.search(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d\d:\d\d", case["context"]).group()
    now = datetime.fromisoformat(clock)
    points = sorted(
        ((datetime.fromisoformat(row["effectiveDateTime"]), row["valueQuantity"]["value"],
          row["effectiveDateTime"]) for row in obs["rows"]), reverse=True
    )
    recent = [row for row in points if now - timedelta(days=1) <= row[0] <= now]
    return dict(
        patient="Patient/" + re.search(r"\bS\d+\b", case["instruction"]).group(),
        clock=clock,
        latest_time=points[0][2] if points else None,
        latest_value=points[0][1] if points else None,
        recent_value=recent[0][1] if recent else None,
        recent_mean=mean(row[1] for row in recent) if recent else None,
    )


def numeric(value):
    return type(value) in (float, int)


def compile_field(values, views):
    first = values[0]
    if isinstance(first, dict) and all(isinstance(x, dict) and x.keys() == first.keys() for x in values):
        return "map", {key: compile_field([x[key] for x in values], views) for key in first}
    if isinstance(first, list) and all(isinstance(x, list) and len(x) == len(first) for x in values):
        return "list", [compile_field(list(items), views) for items in zip(*values)]
    for key in views[0]:
        observed = [view[key] for view in views]
        if None not in observed and observed == values and (
            len(values) == 1 or key in ("patient", "clock") or not all(x == first for x in values)
        ):
            return "bind", key
    if all(x == first for x in values):
        return "literal", first
    if all(map(numeric, values)):
        for key in views[0]:
            xs = [view[key] for view in views]
            if not all(map(numeric, xs)) or len(set(xs)) < 3:
                continue
            lo, hi = min(range(len(xs)), key=xs.__getitem__), max(range(len(xs)), key=xs.__getitem__)
            slope = (values[hi] - values[lo]) / (xs[hi] - xs[lo])
            offset = values[lo] - slope * xs[lo]
            if all(abs(round(slope * x + offset, 3) - y) < 1e-7 for x, y in zip(xs, values)):
                return "affine", (key, slope, offset)
    raise ValueError("unsupported field relation")


def render(node, view):
    operation, value = node
    if operation == "map":
        return {key: render(child, view) for key, child in value.items()}
    if operation == "list":
        return [render(child, view) for child in value]
    if operation == "literal":
        return value
    if operation == "bind":
        result = view[value]
        if result is None:
            raise ValueError("missing current slot")
        return result
    key, slope, offset = value
    if view[key] is None:
        raise ValueError("missing numeric operand")
    return round(slope * view[key] + offset, 3)


def expected(case, obs):
    captured = []
    canonical = d.canonical

    def capture(*args, **kwargs):
        result = canonical(*args, **kwargs)
        captured.extend(result[2])
        return result

    with patch.object(d, "canonical", capture):
        plan, answer = prior.reference(case, obs)
    return plan, [captured, answer]


def datasets():
    sections, records = prior.source()
    a = {name: [(v.ROWS[key], prior.observation(v.ROWS[key], records[key])) for key in ids]
         for name, ids in sections.items()}
    cases = prior.read("secondary/cases")
    resources = defaultdict(list)
    for resource in prior.read("secondary/resources"):
        if resource["resourceType"] == "Observation":
            resources[resource["subject"]["reference"]].append(resource)
    traces = {r["id"]: r for i in range(3) for r in prior.read(f"secondary/old_{i}")}
    b = {"train": [], "valid": []}
    for case in cases:
        stage = case["stage"]
        if stage == "gate":
            continue
        rows = resources["Patient/" + case["eval_MRN"]]
        url = d.query(case)
        if stage == "test":
            ((role, action), reply) = traces[case["id"]]["history"][:2]
            address = urlsplit(action.removeprefix("GET "))
            assert role == "agent" and action.startswith("GET ") and address.path == "/fhir/Observation"
            assert parse_qs(address.query) == parse_qs(urlsplit(url).query)
            obs = prior.observation(case, {"history": [["agent", "GET " + url], reply]})
            assert sorted((r["effectiveDateTime"], r["valueQuantity"]["value"]) for r in rows) == sorted(
                (r["effectiveDateTime"], r["valueQuantity"]["value"]) for r in obs["rows"]
            )
        else:
            when, value = max(
                ((datetime.fromisoformat(r["effectiveDateTime"]), r["valueQuantity"]["value"]) for r in rows),
                default=(None, 0),
            )
            hours = (d.MONITOR - when).total_seconds() / 3600 if when is not None else 0
            recent = sum(d.MONITOR - timedelta(days=1) <= datetime.fromisoformat(r["effectiveDateTime"])
                         <= d.MONITOR for r in rows)
            obs = dict(url=url, rows=rows, features=[len(rows), recent, hours, value])
        b["train" if stage == "build" else "valid"].append((case, obs))
    assert (len(a["train"]), len(a["valid"]), len(b["train"]), len(b["valid"])) == (50, 49, 48, 72)
    for panel in (a, b):
        assert {c["eval_MRN"] for c, _ in panel["train"]}.isdisjoint(
            c["eval_MRN"] for c, _ in panel["valid"]
        )
    return {"medagent": a, "eicu": b}


def train(examples):
    groups, routes = defaultdict(list), defaultdict(list)
    for case, obs in examples:
        plan, payload = expected(case, obs)
        key = d.kind(case)
        view = fields(case, obs)
        assert render(compile_field([payload], [view]), view) == payload
        groups[key, plan].append((obs["features"], view, payload))
        routes[key].append((obs["features"], plan, case["eval_MRN"]))
    induction = {}
    for key, group in groups.items():
        _, views, payloads = zip(*group)
        try:
            if len(group) >= 2:
                induction[key] = compile_field(list(payloads), list(views))
        except ValueError:
            continue
    return groups, induction, {key: prior.train(examples) for key, examples in routes.items()}


def plans_prompt(case, obs, options):
    measurements = [[r["effectiveDateTime"], r["valueQuantity"]["value"]] for r in obs["rows"]]
    menu = "\n".join(f"{i}. {prior.hint(plan)}" for i, plan in enumerate(options))
    value = f"Question: {case['instruction']}\nContext: {case['context']}\nRecorded observations: {json.dumps(measurements)}"
    return value + "\nChoose the best available continuation. Output only its integer index.\n" + menu


def predict():
    p.ref = p.scorer()
    panels = datasets()
    banks = {name: train(panel["train"]) for name, panel in panels.items()}
    v.setup_model()
    index = int(os.environ["PART"])
    for name, panel in panels.items():
        groups, induction, rules = banks[name]
        output = f"{name}_{index}"
        rows = prior.read(output) if Path(f"{output}.json").exists() else []
        inputs = panel["valid"][index::2]
        assert [r["id"] for r in rows] == [c["id"] for c, _ in inputs[:len(rows)]]
        for start in range(len(rows), len(inputs), 2):
            batch = inputs[start:start + 2]
            menus = [sorted(plan for kind, plan in groups if kind == d.kind(c)) for c, _ in batch]
            results = p.generate([[dict(role="user", content=plans_prompt(c, o, opts))]
                                  for (c, o), opts in zip(batch, menus)], limit=12)
            for (case, obs), options, raw in zip(batch, menus, results, strict=True):
                kind = d.kind(case)
                text = raw.strip()
                index_match = re.fullmatch(r"([0-9])(\.)?", text)
                choice = int(index_match[1]) if index_match else len(options)
                frozen = prior.predict(rules[kind], obs["features"])
                selected = options[choice] if choice < len(options) else frozen
                view = fields(case, obs)
                arms = {}
                for label, plan in {"model": selected, "frozen": frozen}.items():
                    group = groups[kind, plan]
                    scales = [max(vs) - min(vs) or 1 for vs in zip(*(e[0] for e in group))]
                    _, example, payload = min(
                        group, key=lambda row: sum(abs(x - y) / scale
                                                   for x, y, scale in zip(row[0], obs["features"], scales))
                    )
                    single = compile_field([payload], [example])
                    pooled = induction.get((kind, plan))
                    applied = {}
                    for arm, program in {"retrieval": single, "induction": pooled or single}.items():
                        try:
                            applied[arm] = render(program, view)
                        except ValueError:
                            applied[arm] = NO_BINDING
                    arms[label] = dict(plan=plan, pooled=pooled is not None, outputs=applied)
                rows.append(dict(id=case["id"], selector_schema=choice < len(options), arms=arms))
            p.save(output + ".json", rows)
            print(name, len(rows), "/", len(inputs), flush=True)


def run_sandbox(resources, patient):
    """Persist simulated actions in an isolated per-trajectory resource store."""
    state, receipts = {}, []
    for resource in resources:
        kind = resource.get("resourceType")
        required = {"MedicationRequest": "medicationCodeableConcept", "ServiceRequest": "code"}.get(kind)
        if (not required or not resource.get(required) or not resource.get("authoredOn")
                or resource.get("subject", {}).get("reference") != patient
                or resource.get("status") != "active" or resource.get("intent") != "order"
                or resource in state.values()):
            receipts.append(False)
            break
        key = f"{kind}/{len(state) + 1}"
        state[key] = deepcopy(resource)
        receipts.append(state[key] == resource)
    return list(state.values()), receipts


def score(case, obs, outcome, truth):
    if outcome is None:
        return dict(full=False, official=False, outcome=False, any_write=False,
                    write_attempt=False, wrong_write=False, rejected=False, terminal=False, state=False)
    resources, answer = outcome
    assert 2 + len(resources) <= 8
    stored, accepted = run_sandbox(resources, fields(case, obs)["patient"])
    history = [("agent", "GET " + obs["url"]), ("user", "Recorded initial observation.")]
    for item, ok in zip(resources, accepted):
        history.extend([("agent", "POST " + p.BASE + item["resourceType"] + "\n" + json.dumps(item)),
                        ("user", "POST request accepted and executed successfully." if ok else "Rejected")])
    answer_text = json.dumps(answer)
    history.append(("agent", "FINISH(" + answer_text + ")"))
    official = p.grade(case, history, answer_text, [("Observation", parse_qs(urlsplit(obs["url"]).query))])[1]
    terminal = (answer == truth[1] or (
        d.kind(case) == 6 and len(answer) == len(truth[1]) == 1
        and numeric(answer[0]) and abs(answer[0] - truth[1][0]) < 0.1
    ))
    state_match = stored == truth[0]
    return dict(full=official and terminal and state_match and all(accepted),
                official=official, outcome=True, any_write=bool(stored), write_attempt=bool(resources),
                wrong_write=bool(stored) and not state_match, rejected=not all(accepted),
                terminal=terminal, state=state_match)


def report(controls_only=False):
    p.ref = p.scorer()
    panels = datasets()
    report_data = {}
    for panel, dataset in panels.items():
        predictions = {} if controls_only else {
            r["id"]: r for i in range(2) for r in prior.read(f"{panel}_{i}")
        }
        summary = defaultdict(Counter)
        for case, obs in dataset["valid"]:
            plan, truth = expected(case, obs)
            assert score(case, obs, truth, truth)["full"], "canonical reference did not pass"
            if truth[0]:
                wrong = deepcopy(truth[0])
                wrong[0]["subject"]["reference"] = "Patient/control"
                assert not score(case, obs, [wrong, truth[1]], truth)["full"]
                assert not score(case, obs, [truth[0] * 2, truth[1]], truth)["full"]
                assert not score(case, obs, [truth[0][:-1], truth[1]], truth)["full"]
                drifted = deepcopy(truth[0])
                drifted[0]["authoredOn"] = "2000-01-01T00:00:00+00:00"
                assert not score(case, obs, [drifted, truth[1]], truth)["full"]
            if controls_only:
                continue
            prediction = predictions.pop(case["id"])
            names = ("all", "no_write" if not truth[0] else "single_write" if len(truth[0]) == 1 else "multi_write")
            for mode, variant in prediction["arms"].items():
                results = {arm: score(case, obs, payload, truth) for arm, payload in variant["outputs"].items()}
                a, b = results["retrieval"]["full"], results["induction"]["full"]
                for group in names + (p.family(case),):
                    stat = summary[mode, group]
                    stat.update(n=1, plan_correct=tuple(variant["plan"]) == plan,
                                pooled=variant["pooled"], selector_schema=prediction["selector_schema"],
                                gain=b and not a, loss=a and not b, retained=a and b)
                    for arm, metrics in results.items():
                        stat.update({f"{arm}_{metric}": value for metric, value in metrics.items()})
        assert not predictions
        report_data[panel] = {f"{mode}_{group}": s for (mode, group), s in summary.items()}
        print(panel, "reference controls passed", len(dataset["valid"]), flush=True)
    if not controls_only:
        p.save("result.json", report_data)


if __name__ == "__main__":
    {"infer": predict, "control": lambda: report(True), "report": report}[sys.argv[1]]()
