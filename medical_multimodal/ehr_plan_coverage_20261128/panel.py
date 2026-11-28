"""Freeze a patient-disjoint follow-up and derive valid counterfactual page sizes."""
import json
from collections import Counter
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent / "ehr_plan_rl_20261128"))
from environment import needs_more
from prepare import freeze, QUESTIONS

SEED = 26112871
VARIANTS = {
    "mean": (*QUESTIONS["mean"],
             "Over a 24-hour lookback, calculate the average blood glucose in mg/dL.",
             "Compute the arithmetic mean of glucose measurements from the previous 24 hours, in mg/dL.",
             "What is the average glucose level in mg/dL for the preceding 24-hour window?"),
    "latest": (*QUESTIONS["latest"],
               "Report the newest available blood glucose measurement in mg/dL, with no age limit.",
               "Give the most recent glucose reading in mg/dL, no matter when it was recorded.",
               "Find the newest glucose result, in mg/dL, even if the measurement is old."),
}


def alternate_page(case):
    if case["operation"] != "mean" or not case["rows"] or case["rows"][0][0] > 24:
        return None
    target = not needs_more(case)
    candidates = [p for p in (1, 2, 4, 8) if p != case["page"]
                  and needs_more({**case, "page": p}) == target]
    return min(candidates, key=lambda p: (abs(p - case["page"]), p)) if candidates else None


def variants(case, index):
    questions = VARIANTS[case["operation"]]
    if case["split"] == "train":
        indices = (index % 4, (index + 2) % 4)
    elif case["split"] == "dev":
        indices = ((index + 3) % 4, index % 4)
    else:
        indices = (4, 5)
    return tuple({**case, "question": questions[i]} for i in indices)


def training_panels(rows):
    ordinary, contrasted = [], []
    for i, case in enumerate(rows):
        first, second = variants(case, i)
        ordinary.extend((first, second))
        other = alternate_page(second)
        contrasted.extend((first, second if other is None else {**second, "page": other}))
    return ordinary, contrasted


def main():
    old = []
    for split in ("train", "dev", "test"):
        old.extend(json.loads(Path(f"i0/{split}.json").read_text()))
    old_subjects = {row["subject_id"] for row in old}
    assert len(old_subjects) == 224
    panel = freeze(Path("i0/i1"), json.loads(Path("i0/i2").read_text()),
                   additional_subjects=old_subjects, seed=SEED)
    assert not old_subjects.intersection(row["subject_id"] for row in panel)
    composition = {"excluded_history": 164 + len(old_subjects), "splits": {}}
    for split in ("train", "dev", "test"):
        rows = [row for row in panel if row["split"] == split]
        Path(split + ".json").write_text(json.dumps(rows, allow_nan=False))
        flips = Counter("adds_need" if not needs_more(row) else "removes_need"
                        for row in rows if alternate_page(row) is not None)
        composition["splits"][split] = dict(n=len(rows), strata=dict(Counter(row["stratum"] for row in rows)),
                                             potential_page_flips=dict(flips))
    assert all(composition["splits"][split]["potential_page_flips"] for split in ("train", "test"))
    Path("composition.json").write_text(json.dumps(composition, indent=2))
    print(json.dumps(composition))


if __name__ == "__main__":
    main()
