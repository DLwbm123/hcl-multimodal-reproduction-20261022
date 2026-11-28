"""Freeze patient-disjoint VitalDB panels and reconstruct two historical exclusions."""
import csv
import gzip
import json
import math
from pathlib import Path
import random
from collections import Counter, defaultdict

SEED = 26112861
SPLITS = {"train": 128, "dev": 32, "test": 64}
STRATA = ("mean_more", "mean_enough", "mean_missing", "latest_present", "latest_missing")
QUESTIONS = {
    "mean": ("Return the average blood glucose in mg/dL over the past 24 hours.",
             "What is the mean glucose from the last 24 hours, in mg/dL?",
             "Summarize the preceding 24 hours with the arithmetic mean glucose in mg/dL."),
    "latest": ("Return the latest recorded glucose value in mg/dL regardless of age.",
               "What was the most recent blood glucose measurement, in mg/dL?",
               "Report the last recorded glucose result in mg/dL, even if older than one day."),
}


def table(path):
    with path.open("rb") as stream:
        compressed = stream.read(2) == b"\x1f\x8b"
    with (gzip.open if compressed else open)(path, "rt", encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def freeze(data, previous_panel, *, additional_subjects=(), seed=SEED):
    subjects = {}
    for row in table(data / "cases.csv.gz"):
        if not all(row.get(key) for key in ("caseid", "subjectid", "caseend")):
            continue
        case, subject, end = int(row["caseid"]), int(row["subjectid"]), float(row["caseend"])
        if math.isfinite(end) and end > 0 and (subject not in subjects or case < subjects[subject][0]):
            subjects[subject] = (case, end)
    metadata = {case: (subject, end) for subject, (case, end) in subjects.items()}
    measurements = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    for row in table(data / "labs.csv.gz"):
        name = row["name"]
        if name not in ("k", "gluc") or not all(row.get(key) for key in ("caseid", "dt", "result")):
            continue
        case = int(row["caseid"])
        if case not in metadata:
            continue
        try:
            stamp, result = float(row["dt"]), float(row["result"])
        except ValueError:
            continue
        if math.isfinite(stamp) and math.isfinite(result) and stamp <= metadata[case][1]:
            measurements[case][name][stamp].add(result)

    old_pools = defaultdict(list)
    for case in sorted(metadata):
        series = measurements[case]["k"]
        if series and len(series[max(series)]) != 1:
            continue
        group = "missing" if not series else "write" if next(iter(series[max(series)])) < 3.5 else "no_write"
        old_pools[group].append(case)
    rng = random.Random(26112721)
    assert {k: len(v) for k, v in old_pools.items()} == {"write": 889, "no_write": 3992, "missing": 1209}
    excluded = set()
    for group in ("write", "no_write", "missing"):
        rng.shuffle(old_pools[group])
        excluded.update(old_pools[group][:40])

    rows_by_case = {}
    for case in sorted(set(metadata) - excluded):
        series = measurements[case]["gluc"]
        if any(len(values) != 1 for values in series.values()):
            continue
        rows_by_case[case] = [[round((metadata[case][1] - stamp) / 3600, 4), next(iter(series[stamp]))]
                              for stamp in sorted(series, reverse=True)]
    prior = defaultdict(list)
    for case, rows in rows_by_case.items():
        active = sum(age <= 24 for age, _ in rows)
        prior["partial" if active > 4 else "complete" if active else "empty"].append(case)
    rng = random.Random(26112737)
    for pool in prior.values():
        rng.shuffle(pool)
    assert previous_panel
    for i, row in enumerate(previous_panel):
        assert row["id"] == "t" + str(i)
        case = prior[row["kind"]].pop()
        assert rows_by_case[case] == row["rows"], "historical panel reconstruction disagrees"
        excluded.add(case)
    assert len(excluded) == 164

    rng = random.Random(seed)
    extra = set(additional_subjects)
    eligible = [case for case in rows_by_case if case not in excluded and metadata[case][0] not in extra]
    eligible.sort()
    rng.shuffle(eligible)
    pools = defaultdict(list)
    for case in eligible:
        rows = rows_by_case[case]
        if len({age for age, _ in rows}) != len(rows):
            continue
        page = rng.choice((1, 2, 4))
        visible = rows[:page]
        active = any(age <= 24 for age, _ in rows)
        if not active:
            strata = ["mean_missing", "latest_present" if rows else "latest_missing"]
        elif len(rows) > page and visible[-1][0] < 24:
            strata = ["mean_more", "latest_present"]
        else:
            strata = ["mean_enough", "latest_present"]
        for stratum in strata:
            pools[stratum].append((case, page))
    chosen = set()
    panel = []
    for split, size in SPLITS.items():
        for i in range(size):
            stratum = STRATA[i % len(STRATA)]
            while pools[stratum] and pools[stratum][-1][0] in chosen:
                pools[stratum].pop()
            if not pools[stratum]:
                raise ValueError(f"insufficient distinct patients in {stratum}")
            case, page = pools[stratum].pop()
            chosen.add(case)
            operation = stratum.split("_")[0]
            questions = QUESTIONS[operation]
            variant = 2 if split == "test" else rng.randrange(2)
            panel.append(dict(subject_id=metadata[case][0], case_id=case, source="VitalDB",
                              split=split, stratum=stratum, operation=operation, page=page,
                              rows=rows_by_case[case], question=questions[variant]))
    assert len({c["subject_id"] for c in panel}) == sum(SPLITS.values())
    return panel


if __name__ == "__main__":
    panel = freeze(Path("i1"), json.loads(Path("i2").read_text()))
    for split in SPLITS:
        rows = [c for c in panel if c["split"] == split]
        Path(split + ".json").write_text(json.dumps(rows, allow_nan=False))
    counts = {split: dict(Counter(c["stratum"] for c in panel if c["split"] == split))
              for split in SPLITS}
    Path("composition.json").write_text(json.dumps({"excluded_prior": 164, "groups": counts}, indent=2))
