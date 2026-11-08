"""Retrospective source-only qualification. Never opens a benchmark test file.

Inputs are private historical artifacts. Public output is an explicit aggregate
allowlist. No inference, clinical relabeling, or automatic experience promotion.
"""
import ast
from collections import Counter
import json
from pathlib import Path
import re


def read(path):
    return json.loads(Path(path).read_text())


def response_contract(result):
    try:
        parsed = ast.literal_eval(result)
        payload = parsed[0]
        text = payload.get("response", str(payload)) if isinstance(payload, dict) else str(payload)
    except (ValueError, SyntaxError, TypeError, IndexError):
        return {"format": "unparsed", "clinical_validity": "unknown"}
    if not text.strip():
        kind = "empty"
    elif text.strip().lower() in {"yes", "no", "matched", "not matched"}:
        kind = "bare_verdict_requires_task_contract"
    elif any(tag in text for tag in ("<|ref|>", "<|/ref|>", "<|box|>", "<|/box|>")):
        # The pinned tokenizer uses integer percentages. This verifies syntax,
        # not the location or truth of the finding.
        pattern = r"<\|ref\|>\s*[^<>]+?\s*<\|/ref\|>\s*<\|box\|>\s*\((\d+),(\d+)\),\((\d+),(\d+)\)\s*<\|/box\|>"
        matches = list(re.finditer(pattern, text))
        rest = re.sub(pattern, "", text).strip()
        good = bool(matches) and not rest
        for m in matches:
            x1, y1, x2, y2 = map(int, m.groups())
            good = good and 0 <= x1 < x2 <= 100 and 0 <= y1 < y2 <= 100
        kind = "structured_boxes_syntax_only" if good else "malformed_structured_output"
    else:
        kind = "free_text_unverified"
    return {"format": kind, "clinical_validity": "unknown"}


def audit(work, output):
    work, output = Path(work), Path(output)
    old = work / "medrax_learning_qualification_v9/private"
    raw = read(work / "medrax_task_memory_v16/private/states.json")["H_B"]["raw"]
    inputs = {r["id"]: r for r in read(old / "inputs.json") if r["role"] == "train"}
    episodes = read(old / "episodes.json")
    primary = {i: e for i, e in enumerate(episodes) if e["state"] == "H0_train"}
    concerns = {r["id"]: r for r in read(old / "source_review/REVIEW.private.json")["rows"]}
    tokens = {r["event_index"]: r for r in read(output / "private/token_audit.json")["rows"]}
    assert len(raw) == len(inputs) == len(primary) == 48
    source_ids = {r["internal_source"]["task_id"] for r in raw}
    assert source_ids == set(inputs) == {e["id"] for e in primary.values()}
    assert all(r["internal_source"]["split"] == "train" for r in raw)
    events = [json.loads(s) for s in (old / "events.jsonl").read_text().splitlines()]
    pending, calls = {}, []
    for index, event in enumerate(events):
        episode = event.get("episode")
        if episode not in primary:
            continue
        kind = event.get("kind")
        key = (episode, event.get("name"))
        if kind == "tool_started":
            assert key not in pending, "Unexpected overlapping tool calls"
            pending[key] = (index, event)
        elif kind == "tool_finished":
            assert key in pending, "Unmatched completion must be investigated"
            start_index, start = pending.pop(key)
            args = start["args"]
            paths = args.get("image_paths", [args.get("image_path")])
            qid = primary[episode]["id"]
            allowed = set(inputs[qid]["images"])
            row = {"source_id": qid, "episode": episode, "event_index": start_index,
                   "tool": event["name"], "status": event["status"],
                   "argument_images": len(paths),
                   "image_binding_pass": bool(paths) and all(p in allowed for p in paths),
                   **response_contract(event["result"])}
            if event["name"] == "chest_xray_expert":
                row["token_contract"] = tokens[start_index]
                assert row["argument_images"] == tokens[start_index]["images"]
            calls.append(row)
    assert not pending
    assert len([c for c in calls if c["tool"] == "chest_xray_expert"]) == len(tokens)
    source_rows = []
    for r in raw:
        qid = r["internal_source"]["task_id"]
        own = [c for c in calls if c["source_id"] == qid]
        source_rows.append({"id": qid, "stage": inputs[qid]["stage"],
            "benchmark_correct": r["response"] == r["feedback"],
            "input_image_count": len(inputs[qid]["images"]),
            "input_matches_memory": inputs[qid]["images"] == r["images"],
            "supervision": "pending_adjudication" if qid in concerns else "benchmark_only_not_independently_adjudicated",
            "prior_concern": concerns.get(qid),
            "tool_calls": own,
            "paired_action_utility": "not_established",
            "callable_corrective_experience": False,
            "next_action": "resolve_supervision_before_clinical_learning" if qid in concerns else "qualify_observations_before_action_pairing"})
    chex = [c for c in calls if c["tool"] == "chest_xray_expert"]
    groups = []
    for n in sorted({c["argument_images"] for c in chex}):
        group = [c for c in chex if c["argument_images"] == n]
        lens = [c["token_contract"]["input_tokens"] for c in group]
        groups.append({"images_per_call": n, "calls": len(group),
                       "input_tokens_min": min(lens), "input_tokens_max": max(lens),
                       "input_exceeds_declared_context": sum(c["token_contract"]["input_exceeds_context"] for c in group),
                       "response_formats": dict(Counter(c["format"] for c in group))})
    summary = {"type": "retrospective_development_diagnostic", "sources": len(source_rows),
        "benchmark_correct": sum(r["benchmark_correct"] for r in source_rows),
        "unresolved_supervision": len(concerns),
        "source_memory_image_mismatches": sum(not r["input_matches_memory"] for r in source_rows),
        "source_image_references": sum(r["input_image_count"] for r in source_rows),
        "primary_tool_calls": len(calls), "tools": dict(Counter(c["tool"] for c in calls)),
        "execution_statuses": dict(Counter(c["status"] for c in calls)),
        "image_binding_failures": sum(not c["image_binding_pass"] for c in calls),
        "chexagent_declared_context": 2048, "chexagent_by_image_count": groups,
        "paired_verified_corrective_experiences": 0,
        "no_evidence_is_not_proof_of_no_possible_utility": True,
        "historical_rechecks_excluded_from_primary": sum(e["role"] == "train" and e["state"] != "H0_train" for e in episodes),
        "new_inference_calls": 0, "confirmation_used": False,
        "privacy_pixel_screening": "not_performed_no_images_sent_to_models",
        "clinical_or_learning_gain_established": False}
    (output / "private/source_qualification.json").write_text(json.dumps(source_rows, ensure_ascii=False, indent=2)+"\n")
    (output / "private/call_qualification.json").write_text(json.dumps(calls, ensure_ascii=False, indent=2)+"\n")
    (output / "AGGREGATE_REPORT.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
    return summary


if __name__ == "__main__":
    import os
    print(json.dumps(audit(os.environ["SOURCE_WORK"], os.environ["AUDIT_OUT"]), indent=2))
