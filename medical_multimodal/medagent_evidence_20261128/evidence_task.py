"""Two-source, fixed-budget query task over recorded Observation results."""
from collections import Counter
from datetime import datetime, timezone
import json
import math
import re

ACTIONS = (("lab",), ("vital",))
NAMES = tuple(action[0] for action in ACTIONS)
CUTOFF = "2023-11-13T10:15:00+00:00"
OFFSETS = (0, 24, 48, 72)
TARGET_HOURS = 24
HISTORY_HOURS = 48
WINDOW = TARGET_HOURS * 3600
HISTORY = HISTORY_HOURS * 3600
CODE = {"lab": ("GLU",), "vital": ("SpO2", "SPO2", "2708-6", "59408-5")}


def timestamp(raw):
    t = datetime.fromisoformat(raw.replace("Z","+00:00"))
    if t.tzinfo is None:
        raise ValueError("Observation timestamp lacks a time zone")
    return t.astimezone(timezone.utc).timestamp()


def extract(row, name, patient):
    """Accept strictly patient-bound numeric readings of a documented source code."""
    if row.get("resourceType") != "Observation" or row.get("subject", {}).get("reference") != "Patient/" + patient:
        raise ValueError("unexpected resource or patient in FHIR response")
    code = row.get("code", {})
    codes = {item.get("code") for item in code.get("coding", ())} | {code.get("text")}
    if not codes.intersection(CODE[name]):
        return None
    if row.get("status") in ("entered-in-error", "cancelled"):
        return None
    q = row.get("valueQuantity", {})
    value = q.get("value")
    unit = q.get("unit") or q.get("code") or ""
    if name == "vital" and value is None:
        match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*%?\s*", row.get("valueString", ""))
        if match:
            value, unit = float(match[1]), "%"
    raw_date = row.get("effectiveDateTime")
    if not raw_date or type(value) not in (float, int) or not math.isfinite(value):
        return None
    return dict(id=str(row["id"]), at=timestamp(raw_date), value=value,
                unit=unit)


def episode(patient, cutoff, offset, sources):
    previous = {}
    for name in NAMES:
        prior = [item["at"] for item in sources[name] if cutoff-WINDOW-HISTORY <= item["at"] < cutoff-WINDOW]
        previous[name] = dict(count=len(prior), hours_since_prior=(cutoff-max(prior))//3600 if prior else None)
    return dict(patient=patient, cutoff=cutoff, offset=offset, evidence=sources, index=previous)


def prompt(case):
    return (
        "In a retrospective chart review, retrieve one documented numeric result from the last 24 hours "
        "ending at the cutoff. Either a glucose lab result or a pulse-oximetry vital result is acceptable. "
        f"The cutoff is {case['offset']} hours before the reference review time. "
        "The prior-history index describes the 48 hours strictly before the target window. "
        "It does not say which source contains a result in the target window. "
        "You can call exactly one source. The tool returns the most recent eligible record. "
        'Respond with exactly {"actions":["lab"]} or {"actions":["vital"]}.\n'
        "Prior-history index: " + json.dumps(case["index"], separators=(",", ":"))
    )


def use(case, name):
    if name not in NAMES:
        raise ValueError("action outside tool budget")
    cutoff = case["cutoff"]
    eligible = [row for row in case["evidence"][name] if cutoff-WINDOW <= row["at"] <= cutoff]
    return max(eligible, key=lambda row: (row["at"], row["id"]), default=None)


def success(case, name):
    # Verify the returned ID, numeric payload and target interval against the extracted records.
    observed = use(case, name)
    if observed is None:
        return False
    return any(item == observed and 0 <= case["cutoff"]-item["at"] <= WINDOW
               and math.isfinite(item["value"]) for item in case["evidence"][name])


def index_rule(case, fallback):
    rank = {}
    for source, data in case["index"].items():
        recency = data["hours_since_prior"]
        rank[source] = (recency is not None, -recency if recency is not None else 0, data["count"], source == fallback)
    return max(rank, key=rank.get)


def score_policy(cases, chooser):
    decisions = [chooser(c) for c in cases]
    correct = [success(c, a) for c, a in zip(cases, decisions, strict=True)]
    return correct, dict(success=sum(correct), episodes=len(correct), picks=dict(Counter(decisions)))


def self_check():
    anchor = timestamp(CUTOFF)
    source = dict(lab=[dict(id="a", at=anchor-WINDOW-1, value=1, unit="u")],
                  vital=[dict(id="b", at=anchor-WINDOW, value=2, unit="u"),
                         dict(id="c", at=anchor+1, value=3, unit="u")])
    case = episode("private", anchor, 0, source)
    assert use(case, "lab") is None and use(case, "vital")["id"] == "b"
    assert success(case, "vital") and not success(case, "lab")
    assert case["index"]["lab"]["count"] == 1 and case["index"]["vital"]["count"] == 0
    assert "private" not in prompt(case) and "evidence" not in prompt(case)
    vital = dict(resourceType="Observation", subject=dict(reference="Patient/x"),
                 code=dict(text="SpO2"), effectiveDateTime=CUTOFF, valueString="97%", id="fixture")
    assert extract(vital, "vital", "x")["value"] == 97
    assert extract({**vital,"valueString":"missing"}, "vital", "x") is None
    try:
        use(case, "both")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid tool executed")
