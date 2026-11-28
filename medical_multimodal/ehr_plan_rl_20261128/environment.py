"""Read-only EHR plan environment and reward shared by all arms."""
from dataclasses import dataclass
import json
import math
from statistics import mean

PLANS = (("mean",), ("latest",), ("more", "mean"), ("more", "latest"))


def prompt(case):
    visible = case["rows"][:case["page"]]
    receipt = dict(rows=visible, has_more=len(visible) < len(case["rows"]))
    return (
        "Question: " + case["question"] + "\n"
        "The receipt lists [hours before now, glucose in mg/dL], newest first. "
        "Timestamps are distinct. has_more indicates unread older rows. "
        "more reads all remaining rows. mean returns the loaded mean within the last 24 hours, "
        "including the boundary. latest returns the newest loaded measurement, regardless of age. "
        "The final tool returns NA when it has no qualifying records. "
        "Use enough evidence to answer and avoid unnecessary calls. "
        "Output exactly one JSON object from these options: "
        '{"actions":["mean"]}, {"actions":["latest"]}, {"actions":["more","mean"]}, '
        '{"actions":["more","latest"]}.\n'
        "Receipt: " + json.dumps(receipt, separators=(",", ":"))
    )


def parse(output):
    try:
        plan = json.loads(output, object_pairs_hook=lambda pairs: pairs)
    except (ValueError, TypeError):
        return ()
    if isinstance(plan, list) and len(plan) == 1 and isinstance(plan[0], tuple):
        key, actions = plan[0]
        if key == "actions" and isinstance(actions, list) and tuple(actions) in PLANS:
            return tuple(actions)
    return ()


def encode(plan):
    return json.dumps({"actions": plan}, separators=(",", ":"))


def needs_more(case):
    visible = case["rows"][:case["page"]]
    return (case["operation"] == "mean" and len(visible) < len(case["rows"])
            and visible[-1][0] < 24)


def rule(case):
    return (("more",) if needs_more(case) else ()) + (case["operation"],)


@dataclass(frozen=True)
class Score:
    legal: bool = False
    correct: bool = False
    evidence: bool = False
    unnecessary: bool = False
    premature: bool = False
    calls: int = 0

    @property
    def success(self):
        return self.legal and self.correct and self.evidence

    @property
    def reward(self):
        return 1 - 0.02 * self.calls if self.success else -1.0


def execute(case, plan):
    if plan not in PLANS:
        return Score()
    rows = case["rows"]
    loaded = rows if plan[0] == "more" else rows[:case["page"]]
    if plan[-1] == "latest":
        answer = loaded[0][1] if loaded else None
    else:
        values = [value for age, value in loaded if age <= 24]
        answer = mean(values) if values else None

    if case["operation"] == "latest":
        expected = rows[0][1] if rows else None
    else:
        values = [value for age, value in rows if age <= 24]
        expected = math.fsum(values) / len(values) if values else None
    matches = answer is expected if None in (answer, expected) else math.isclose(
        answer, expected, rel_tol=0.0, abs_tol=1e-6
    )
    more = plan[0] == "more"
    required = needs_more(case)
    return Score(legal=True, correct=plan[-1] == case["operation"] and matches,
                 evidence=not required or more, unnecessary=more and not required,
                 premature=required and not more, calls=len(plan))


def self_check():
    cases = [
        ({"operation": "mean", "page": 1, "rows": [[1, 70], [23, 70]]}, (False, False, True, False)),
        ({"operation": "mean", "page": 1, "rows": [[24, 80], [25, 90]]}, (True, False, True, False)),
        ({"operation": "mean", "page": 4, "rows": [[25, 90]]}, (True, False, True, False)),
        ({"operation": "latest", "page": 1, "rows": [[25, 90], [28, 90]]}, (False, True, False, True)),
        ({"operation": "latest", "page": 2, "rows": []}, (False, True, False, True)),
    ]
    for case, expected in cases:
        assert tuple(execute(case, plan).success for plan in PLANS) == expected
        assert execute(case, rule(case)).reward == max(execute(case, plan).reward for plan in PLANS)
    assert execute(cases[0][0], ("mean",)).correct
    assert execute(cases[0][0], ("mean",)).premature
    assert execute(cases[1][0], ("more", "mean")).unnecessary
    assert parse('{"actions":["mean"],"actions":["latest"]}') == ()
    assert parse('{"actions":["more","more","mean"]}') == ()
    assert parse('[["actions",["mean"]]]') == ()
    assert all(parse(encode(plan)) == plan for plan in PLANS)
