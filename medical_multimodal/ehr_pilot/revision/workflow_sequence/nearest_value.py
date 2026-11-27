"""Check against the original eICU numeric-nearest rule without changing the learned policy."""

import random
import re
from collections import Counter, defaultdict
from statistics import mean
from unittest.mock import patch

import workflow as w

b, p, d, prior = w.b, w.p, w.d, w.old
initial_execute = w.execute


def execute(model, case, obs, method, conditioned=True, overrides=None):
    if method != "retrieval":
        return initial_execute(model, case, obs, method, conditioned, overrides)
    kind = d.kind(case)
    plan = w.select_plan(model, kind, obs, conditioned)
    view = b.fields(case, obs)
    latest = view["latest_value"]
    group = model[0][kind, plan]
    eligible = [item for item in group if (item[1]["latest_value"] is None) == (latest is None)] or group

    def distance(item):
        val = item[1]["latest_value"]
        return abs(val - latest) if b.numeric(val) and b.numeric(latest) else 0

    _, stored, payload = min(eligible, key=distance)
    try:
        return plan, b.render(b.compile_field([payload], [stored]), view)
    except ValueError:
        return plan, None


def estimate(cases, model):
    patient_deltas = defaultdict(list)
    total = Counter()
    write = Counter()
    for case, obs in cases:
        candidates = {name: execute(model, case, obs, name)[1] for name in ("retrieval", "induction")}
        _, truth = b.expected(case, obs)
        a, z = (b.score(case, obs, candidate, truth)["full"] for candidate in candidates.values())
        patient_deltas[case["eval_MRN"]].append(int(z) - int(a))
        for acc in (total, write) if truth[0] else (total,):
            acc.update(n=1, retrieval=a, induction=z, gain=z and not a, loss=a and not z,
                       retained=a and z)
    clusters = list(patient_deltas.values())
    rng = random.Random(26112720)
    differences = sorted(mean(x for c in rng.choices(clusters, k=len(clusters)) for x in c)
                         for i in range(4000))
    return dict(all=total, write=write, clusters=len(clusters),
                ci_95=[differences[100], differences[3899]])


def dispatch_replay(native, eicu, model):
    from t import valid_query

    rows = {(r["panel"], r["id"]): r for i in range(2) for r in prior.read(f"dispatch_{i}")}
    result = {}
    for panel, cases in (("native", native), ("compose", eicu)):
        stat = Counter()
        for case, obs in cases:
            rec = rows[panel, case["id"]]
            action = rec["action"]
            wrapped = re.fullmatch(r"```[a-zA-Z0-9_-]*\s*\n([^\n]+)\n\s*```", action)
            fixed = wrapped[1].strip() if wrapped else action
            correct = rec["kind"] == (d.kind(case) if panel == "native" else 0)
            stat.update(n=1, scope=correct, strict=valid_query(case, action),
                        unwrapped=valid_query(case, fixed), both_after_unwrap=correct and valid_query(case, fixed))
            if panel == "native" and rec["kind"] != 0:
                scoped = dict(case, id=f"task{rec['kind']}_dispatch")
                guess = initial_execute(model, scoped, obs, "induction")[1]
                _, truth = b.expected(case, obs)
                stat.update(induction_with_verified_query=b.score(case, obs, guess, truth)["full"])
        result[panel] = stat
    return result


def main():
    p.ref = p.scorer()
    examples, panels = w.bank_rows()
    model = w.fit(examples)
    eicu = panels["eicu"]["valid"]
    data = {row["id"]: row["observation"] for i in range(2)
            for row in prior.read(f"previous/query_{i}")}
    native = [(case, data[case["id"]]) for case in w.fresh_cases()]
    assert all(obs is not None for _, obs in native)
    with patch.object(w, "execute", execute):
        result = dict(native=estimate(native, model), development=estimate(eicu, model),
                      composition=w.compose(eicu, model))

    def role_matched(model, case, obs, method, conditioned=True, overrides=None):
        # Keep the original task-specific closest-value retrieval for dosage.
        # The mean branch retains the original full-receipt distance.
        policy = execute if d.kind(case) == 9 else initial_execute
        return policy(model, case, obs, method, conditioned, overrides)

    with patch.object(w, "execute", role_matched):
        result["composition_role_matched"] = w.compose(eicu, model)
    result["dispatch_replay"] = dispatch_replay(native, eicu, model)
    earlier = {row["id"] for arm in ("valbase", "valskill") for row in prior.read(f"historical_pilot/{arm}")}
    novel = [(case, obs) for case, obs in native if case["id"] not in earlier]
    assert len(novel) == 15
    result["native_new_questions_on_seen_patients"] = estimate(novel, model)
    result["native_new_questions_original_distance"] = w.summaries(
        novel, model, "historical patients, benchmark question IDs absent from the checked pilot outputs"
    )["totals"]["all"]
    p.save("nearest_value_result.json", result)


if __name__ == "__main__":
    main()
