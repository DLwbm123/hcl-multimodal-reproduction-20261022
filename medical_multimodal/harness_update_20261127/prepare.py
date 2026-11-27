"""Freeze two paired panels before invoking either GPU worker."""
import csv
import gzip
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROLES = {"fit": 12, "gate": 8, "anchor": 8, "test": 16}


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write(panel, cases, composition):
    labels = {r["id"]: r.pop("answer") for r in cases}
    assert len(labels) == sum(ROLES.values())
    assert Counter(r["role"] for r in cases) == ROLES
    for name, obj in (("cases", cases), ("labels", labels), ("split_summary", composition)):
        (panel / (name + ".json")).write_text(json.dumps(obj, ensure_ascii=False))


def image(panel):
    root = Path("i0/medical_transfer_v1/prepared_data")
    historical = Path("i0/agent_visual_20261115")
    members = json.loads((historical / "private_membership.json").read_text())
    manifest = read_lines(root / "audit/roles.jsonl")
    by_id = {row["id"]: row for row in manifest}
    groups, pixels = set(), set()

    def collect(obj):
        if isinstance(obj, dict):
            for key in ("id", "task_id", "internal_sample_id"):
                value = obj.get(key)
                if isinstance(value, str) and value in by_id:
                    ref = by_id[value]
                    groups.add(ref["group_id"])
                    pixels.add(ref["pixel_id"])
            if obj.get("group_id"):
                groups.add(obj["group_id"])
            if obj.get("pixel_id"):
                pixels.add(obj["pixel_id"])
            for value in obj.values():
                collect(value)
        elif isinstance(obj, list):
            for value in obj:
                collect(value)

    files = {Path(path) for path in members["exclusion_sources"]}
    files.update(Path("i0").glob("*/private_membership.json"))
    for path in sorted(files):
        if path.is_file():
            collect(read_lines(path) if path.suffix == ".jsonl" else json.loads(path.read_text()))
    label_rows = {r["id"]: r for r in read_lines(root / "private/visual_qa/h_search.jsonl")}
    visible = read_lines(root / "visible/visual_qa/h_search.jsonl")
    rng = random.Random(26112736)
    rng.shuffle(visible)
    positive_groups = {ref["metadata"]["group_id"] for ref in label_rows.values()
                       if ref["metadata"]["question_type"] == "yes_no_or_presence"
                       and re.match(r"\s*yes\b", str(ref["answer"]), re.I)}
    taken, occupied_groups, occupied_pixels = [], set(), set()
    positives = {"fit": 3, "gate": 2, "test": 3}
    historical_anchors = 0
    for role in ("fit", "gate", "test", "anchor"):
        n = ROLES[role]
        picked = Counter()
        for row in visible:
            ref = label_rows[row["id"]]
            meta = ref["metadata"]
            group, pixel, kind = meta["group_id"], meta["pixel_id"], meta["question_type"]
            if role == "anchor" and kind == "yes_no_or_presence":
                continue
            if role != "anchor" and kind != "yes_no_or_presence":
                continue
            if group in groups | occupied_groups or pixel in pixels | occupied_pixels:
                continue
            if not Path(row["image"]).is_file() or not isinstance(ref["answer"], str) or not ref["answer"].strip():
                continue
            if role == "anchor":
                bucket, answer = kind, ref["answer"]
            else:
                match = re.match(r"\s*(yes|no)\b", ref["answer"], re.I)
                if not match:
                    continue
                bucket = answer = match[1].lower()
                if bucket == "no" and group in positive_groups:
                    continue
                quota = positives[role] if bucket == "yes" else n - positives[role]
                if picked[bucket] >= quota:
                    continue
            taken.append(dict(id="i"+str(len(taken)), role=role, kind=kind,
                              question=row["question"], image=str(Path(row["image"]).resolve()), answer=answer))
            picked[bucket] += 1
            occupied_groups.add(group)
            occupied_pixels.add(pixel)
            if sum(picked.values()) == n:
                break
        if role == "anchor" and sum(picked.values()) < n:
            old = json.loads((historical / "private_learning.json").read_text())["development"]
            for row in old:
                meta = row["metadata"]
                group, pixel, kind = meta["group_id"], meta["pixel_id"], meta["question_type"]
                if (kind == "yes_no_or_presence" or group in occupied_groups or pixel in occupied_pixels
                        or not Path(row["image"]).is_file()):
                    continue
                taken.append(dict(id="i"+str(len(taken)), role=role, kind=kind, question=row["question"],
                                  image=str(Path(row["image"]).resolve()), answer=row["answer"]))
                picked[kind] += 1
                occupied_groups.add(group)
                occupied_pixels.add(pixel)
                historical_anchors += 1
                if sum(picked.values()) == n:
                    break
        assert sum(picked.values()) == n, (role, picked)
    write(panel, taken, dict(source="endoscopy h_search", excluded_groups=len(groups),
          checked_manifests=sum(path.is_file() for path in files), distinct_source_groups=len(occupied_groups),
          historical_anchor_reuse=historical_anchors))


def table(name):
    path = Path("i1") / (name + ".csv.gz")
    with path.open("rb") as fd:
        compressed = fd.read(2) == b"\x1f\x8b"
    with (gzip.open if compressed else open)(path, "rt", encoding="utf-8-sig", newline="") as fd:
        yield from csv.DictReader(fd)


def text(panel):
    subjects = {}
    for row in table("cases"):
        if not all(row.get(k) for k in ("caseid", "subjectid", "caseend")):
            continue
        case, person, end = int(row["caseid"]), int(row["subjectid"]), float(row["caseend"])
        if math.isfinite(end) and end > 0 and (person not in subjects or case < subjects[person][0]):
            subjects[person] = (case, end)
    ends = dict(subjects.values())
    values = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    for row in table("labs"):
        if row["name"] not in ("k", "gluc") or not row["dt"] or not row["result"]:
            continue
        case = int(row["caseid"])
        try:
            stamp, val = float(row["dt"]), float(row["result"])
        except ValueError:
            continue
        if case in ends and math.isfinite(val) and math.isfinite(stamp) and stamp <= ends[case]:
            values[case][row["name"]][stamp].add(val)
    prior_pools = defaultdict(list)
    for case in sorted(ends):
        series = values[case]["k"]
        if series and len(series[max(series)]) != 1:
            continue
        group = "missing" if not series else "write" if next(iter(series[max(series)])) < 3.5 else "no_write"
        prior_pools[group].append(case)
    assert {k: len(v) for k, v in prior_pools.items()} == {"write": 889, "no_write": 3992, "missing": 1209}
    old = set()
    rng = random.Random(26112721)
    for group in ("write", "no_write", "missing"):
        source = prior_pools[group]
        rng.shuffle(source)
        old.update(source[:40])
    assert len(old) == 120
    pools = defaultdict(list)
    for case in sorted(set(ends) - old):
        series = values[case]["gluc"]
        if any(len(v) != 1 for v in series.values()):
            continue
        rows = [[round((ends[case] - stamp) / 3600, 4), next(iter(series[stamp]))]
                for stamp in sorted(series, reverse=True)]
        active = [v for ago, v in rows if ago <= 24]
        kind = "partial" if len(active) > 4 else "complete" if active else "empty"
        pools[kind].append((rows, active))
    rng = random.Random(26112737)
    for group in pools.values():
        rng.shuffle(group)
    cases = []
    for role, n in ROLES.items():
        for i in range(n):
            kind = ("complete", "partial", "empty")[i % 3]
            rows, current = pools[kind].pop()
            query = ("Return the latest recorded glucose result in mg/dL." if role == "anchor"
                     else "Return the mean glucose in mg/dL over the last 24 hours.")
            answer = (rows[0][1] if rows else "NA") if role == "anchor" else (
                sum(current) / len(current) if current else "NA")
            cases.append(dict(id="t"+str(len(cases)), role=role, kind=kind, question=query, rows=rows, answer=answer))
    write(panel, cases, dict(source="VitalDB", distinct_subjects=len(cases), excluded_prior_panel=len(old)))


if __name__ == "__main__":
    image(Path("a"))
    text(Path("b"))
