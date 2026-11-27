"""Sequential proof of acquisition and retention on existing disjoint eICU splits."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import workflow as w

b, p, d = w.b, w.p, w.d


def panel(path):
    root = Path(path)
    cases = json.loads((root / "cases.json").read_text())
    rows = defaultdict(list)
    for resource in json.loads((root / "resources.json").read_text()):
        if resource["resourceType"] == "Observation":
            rows[resource["subject"]["reference"]].append(resource)
    groups = defaultdict(list)
    for case in cases:
        obs = w.observation(case, rows["Patient/" + case["eval_MRN"]])
        groups[case["stage"]].append((case, obs))
    splits = [{c["eval_MRN"] for c, _ in group} for group in groups.values()]
    assert sum(map(len, splits)) == len(set.union(*splits))
    return groups


def shared_medication(new):
    """Obtain the newest program for the single overlapping resource type."""
    key = next(k for k in new[0] if k[0] == 5 and k[1][0] == "MedicationRequest")
    first = new[0][key][0]
    compiled = new[1].get(key) or b.compile_field([first[2]], [first[1]])
    kind, payload = compiled
    resources_kind, resources = payload[0]
    assert kind == resources_kind == "list" and len(resources) == 1
    return resources[0]


def score(cases, predictor):
    values = []
    for case, obs in cases:
        payload = predictor(case, obs)
        _, truth = b.expected(case, obs)
        assert b.score(case, obs, truth, truth)["full"]
        values.append(b.score(case, obs, payload, truth)["full"])
    return values


def main():
    p.ref = p.scorer()
    old, new = panel("secondary"), panel("later")
    assert [len(old[x]) for x in ("build", "gate", "test")] == [48, 24, 72]
    assert [len(new[x]) for x in ("build", "gate", "test")] == [360, 72, 144]
    small = Counter()
    examples = []
    for c, o in new["build"]:
        small[c["group"]] += 1
        if small[c["group"]] <= 16:
            examples.append((c, o))
    assert len(examples) == 48
    a = w.fit(old["build"])
    joint = w.fit(old["build"] + examples)
    recent = shared_medication(joint)

    def payload(model, method, case, obs):
        return w.execute(model, case, obs, method)[1]

    def merged(case, obs):
        candidate = payload(joint, "induction", case, obs)
        if candidate and d.kind(case) == 9 and candidate[0]:
            bound = b.render(recent, b.fields(case, obs))
            return [[bound, *candidate[0][1:]], candidate[1]]
        return candidate

    def scoped(case, obs):
        return payload(joint, "induction", case, obs)

    def previous(case, obs):
        return payload(a, "induction", case, obs)

    def retrieval(case, obs):
        return payload(joint, "retrieval", case, obs)

    correct_a_gate = score(old["gate"], previous)
    candidates = {"shared": merged, "scoped": scoped}
    gate = {}
    for name, predictor in candidates.items():
        old_hits = score(old["gate"], predictor)
        gate[name] = dict(old=sum(old_hits), new=sum(score(new["gate"], predictor)),
                          lost=sum(x and not y for x, y in zip(correct_a_gate, old_hits)))
    unprotected = max(candidates, key=lambda name: gate[name]["new"])
    eligible = [name for name in candidates if gate[name]["lost"] == 0]
    protected = max(eligible, key=lambda name: gate[name]["new"]) if eligible else "prior"

    retrieval_a = score(old["test"], retrieval)
    stage_a = score(old["test"], previous)
    gained_a = [y and not x for x, y in zip(retrieval_a, stage_a)]
    result = dict(a_n=len(old["test"]), b_n=len(new["test"]), a_retrieval=sum(retrieval_a),
                  a_after_a=sum(stage_a), acquired_a=sum(gained_a),
                  b_before=sum(score(new["test"], previous)),
                  b_retrieval=sum(score(new["test"], retrieval)),
                  anchor=gate, selected=dict(unguarded=unprotected, guarded=protected))
    for branch, chosen in (("unguarded", unprotected), ("guarded", protected)):
        predict = candidates.get(chosen, previous)
        now = score(old["test"], predict)
        then = score(new["test"], predict)
        result[branch] = dict(a=sum(now), b=sum(then),
                              old_lost=sum(x and not y for x, y in zip(stage_a, now)),
                              acquired_retained=sum(x and y for x, y in zip(gained_a, now)))
    p.save("retention_result.json", result)


if __name__ == "__main__":
    main()
