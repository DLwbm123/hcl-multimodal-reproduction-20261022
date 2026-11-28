"""Replay verified official MedAgentBench FHIR data without exposing patient IDs."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import random
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit
from urllib.request import build_opener, ProxyHandler

from evidence_task import (CODE, CUTOFF, HISTORY, NAMES, OFFSETS, WINDOW, episode, extract,
                           index_rule, score_policy, self_check, success, timestamp)

URL = "http://127.0.0.1:18489/fhir/Observation"
SEED = 26112891
OPENER = build_opener(ProxyHandler({}))
SIZES = {"train": 54, "dev": 20, "test": 24}


def fetch(patient, name, start, end):
    # Native vital records encode SpO2 as text, so search the vital category.
    fields = {"code":",".join(CODE[name])} if name == "lab" else {"category":"vital-signs"}
    low = datetime.fromtimestamp(start, timezone.utc).isoformat()
    high = datetime.fromtimestamp(end, timezone.utc).isoformat()
    query = urlencode([("patient",patient), ("_count",1000), *fields.items(), ("date","ge"+low), ("date","le"+high)])
    url = URL+"?"+query
    expected = urlsplit(URL)
    rows = {}
    visited = set()
    while url:
        parsed = urlsplit(url)
        if (parsed.scheme, parsed.netloc) != (expected.scheme, expected.netloc):
            raise ValueError("unexpected FHIR pagination target")
        if url in visited or len(visited) >= 16:
            raise ValueError("FHIR pagination failed to terminate")
        visited.add(url)
        with OPENER.open(url, timeout=40) as reply:
            bundle = json.load(reply)
        if bundle.get("resourceType") != "Bundle" or bundle.get("type") != "searchset":
            raise ValueError("expected searchset Bundle")
        for entry in bundle.get("entry", ()):
            if entry.get("search", {}).get("mode") == "outcome":
                if any(issue.get("severity") in ("fatal","error") for issue in entry["resource"].get("issue", ())):
                    raise ValueError("FHIR search error in Bundle")
                continue
            record = extract(entry["resource"], name, patient)
            if record is not None and start <= record["at"] <= end:
                rows[record["id"]] = record
        following = next((link["url"] for link in bundle.get("link", ()) if link.get("relation") == "next"), None)
        url = None
        if following:
            link = urlsplit(following)
            if link.path not in ("/fhir", "/fhir/", expected.path) or "_getpages" not in parse_qs(link.query):
                raise ValueError("unexpected FHIR pagination link")
            # Keep the server-provided cursor, but only send it to the private loopback server.
            url = urlunsplit((expected.scheme, expected.netloc, link.path, link.query, ""))
    return sorted(rows.values(), key=lambda x:(-x["at"], x["id"]))


def collect(patient):
    hi = timestamp(CUTOFF)
    lo = hi - max(OFFSETS)*3600 - WINDOW - HISTORY
    sources = {name:fetch(patient, name, lo, hi) for name in NAMES}
    return [episode(patient, hi-offset*3600, offset, sources) for offset in OFFSETS]


def structure(cases):
    cells = Counter()
    for case in cases:
        lab, vital = (success(case, name) for name in NAMES)
        cells["both" if lab and vital else "lab_only" if lab else "vital_only" if vital else "neither"] += 1
    return dict(cells=dict(cells), patients=len({c["patient"] for c in cases}), episodes=len(cases))


def fit_fixed(cases):
    def best(for_cases):
        return max(NAMES, key=lambda name:sum(success(c,name) for c in for_cases))
    return best(cases), {offset:best([c for c in cases if c["offset"]==offset]) for offset in OFFSETS}


def audit(cases, fixed, schedule):
    scores = {}
    for name, choose in ((n, lambda c, n=n:n) for n in NAMES):
        _, scores[name] = score_policy(cases, choose)
    _, scores["fixed"] = score_policy(cases, lambda c:fixed)
    _, scores["fixed_by_offset"] = score_policy(cases, lambda c:schedule[c["offset"]])
    _, scores["prior_index_rule"] = score_policy(cases, lambda c:index_rule(c,fixed))
    oracle = [any(success(c,n) for n in NAMES) for c in cases]
    scores["oracle"] = dict(success=sum(oracle), episodes=len(cases))
    _, best_schedule = fit_fixed(cases)
    fixed_scores, scores["best_fixed_hindsight_by_offset"] = score_policy(cases, lambda c:best_schedule[c["offset"]])
    gap = sum(oracle)-sum(fixed_scores)
    gain_patients = len({c["patient"] for c,a,b in zip(cases,fixed_scores,oracle,strict=True) if b and not a})
    return {**structure(cases),"policies":scores,"oracle_minus_best_fixed":gap,
            "patients_with_oracle_gain":gain_patients}


def main(path):
    self_check()
    tasks = json.loads(Path(path).read_text())
    patients = sorted({str(t["eval_MRN"]) for t in tasks if t.get("eval_MRN")})
    assert len(tasks)==300 and len(patients)==98, (len(tasks),len(patients))
    random.Random(SEED).shuffle(patients)
    offset = 0
    split = {}
    for name,n in SIZES.items():
        split[name] = patients[offset:offset+n]
        offset += n
    assert offset == len(patients)
    with ThreadPoolExecutor(max_workers=5) as workers:
        train = [case for group in workers.map(collect,split["train"]) for case in group]
        dev = [case for group in workers.map(collect,split["dev"]) for case in group]
        fixed, schedule = fit_fixed(train)
        dev_score = audit(dev,fixed,schedule)
        gap = dev_score["oracle_minus_best_fixed"]
        # Require a real, non-single-patient gap against the stronger fixed comparator.
        gate = gap/len(dev)>=0.05 and dev_score["patients_with_oracle_gain"]>=2
        result = dict(train=structure(train),dev=dev_score,fixed=fixed,
                      fixed_by_offset=schedule,passes_training_gate=gate)
        print(json.dumps(result),flush=True)
        test = [case for group in workers.map(collect,split["test"]) for case in group]
    for name,rows in (("train",train),("dev",dev),("test",test)):
        Path(name+".json").write_text(json.dumps(rows, allow_nan=False))
    Path("gate.json").write_text(json.dumps(result,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("task_file")
    main(parser.parse_args().task_file)
