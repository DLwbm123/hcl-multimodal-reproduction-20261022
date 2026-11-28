"""Freeze real patients with both incomplete and sufficient evidence pages."""
from collections import Counter
import json
from pathlib import Path
import random
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent / "ehr_plan_rl_20261128"))
from prepare import eligible_cases

SEED = 26112881
SIZES = {"train": 64, "dev": 32, "test": 64}
FORMS = {
    "train": (
        "Return the average glucose in mg/dL using all readings from the past 24 hours.",
        "What is the mean blood glucose in the last 24 hours? Report it in mg/dL.",
        "Calculate the 24-hour arithmetic mean of the patient's glucose measurements in mg/dL.",
        "Use the glucose records from the previous 24 hours and report their average, in mg/dL.",
    ),
    "dev": (
        "Within the one-day lookback ending now, what was the average glucose level in mg/dL?",
        "I need one number in mg/dL: the arithmetic mean across glucose measurements no more than 24 hours old.",
        "Summarize this patient's glucose, in mg/dL, by taking the mean of the readings in the 24 hours up to now.",
    ),
    "test": (
        "What was this patient's average glucose, in mg/dL, for the preceding 24-hour window?",
        "Give the arithmetic average in mg/dL of the glucose measurements recorded no earlier than 24 hours ago.",
        "For the glucose report, use a window ending now and starting 24 hours earlier. Report the mean in mg/dL.",
        "Consider every glucose reading within 24 hours of now. What single arithmetic mean in mg/dL represents them?",
    ),
}


def view(patient, state, question):
    return dict(rows=patient["rows"], page=patient[state], operation="mean", question=question)


def main():
    history = (Path("i0/i0"), Path("i0"))
    old = [{row["subject_id"] for split in SIZES
            for row in json.loads((root / (split + ".json")).read_text())} for root in history]
    assert all(len(ids) == 224 for ids in old) and not old[0] & old[1]
    assert len({q for forms in FORMS.values() for q in forms}) == sum(map(len, FORMS.values()))
    candidates = eligible_cases(history[0] / "i1", json.loads((history[0] / "i2").read_text()), old[0] | old[1])
    rng = random.Random(SEED)
    pools = {True: [], False: []}
    for case, (subject, rows) in sorted(candidates.items()):
        if len({age for age, _ in rows}) != len(rows):
            continue
        in_window = sum(age <= 24 for age, _ in rows)
        if in_window < 2:
            continue
        full = in_window if rows[in_window - 1][0] == 24 or in_window == len(rows) else in_window + 1
        if full > 16:
            continue
        low = rng.randint(1, min(4, in_window - 1))
        pools[full < len(rows)].append(dict(case_id=case, subject_id=subject, rows=rows,
                                            insufficient=low, sufficient=full))
    available = {str(flag): len(rows) for flag, rows in pools.items()}
    assert sum(available.values()) >= sum(SIZES.values()), available
    for rows in pools.values():
        rng.shuffle(rows)
    seen = set()
    summary = {"excluded_history": 164 + len(old[0] | old[1]), "available": available, "splits": {}}
    for split, n in SIZES.items():
        rows = []
        for i in range(n):
            flag = i % 2 == 0
            pool = pools[flag] or pools[not flag]
            patient = pool.pop()
            subject = patient["subject_id"]
            assert subject not in seen and subject not in old[0] and subject not in old[1]
            seen.add(subject)
            rows.append(patient)
        Path(split + ".json").write_text(json.dumps(rows, allow_nan=False))
        summary["splits"][split] = dict(patients=n, populated_views=n * 2 * len(FORMS[split]),
                                        sufficient_has_more=sum(r["sufficient"] < len(r["rows"]) for r in rows),
                                        insufficient_page_sizes=dict(Counter(r["insufficient"] for r in rows)))
    Path("composition.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
