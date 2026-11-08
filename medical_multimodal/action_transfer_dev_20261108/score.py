"""Fixed output-contract control for the source-disjoint development challenge."""
import importlib.util
from pathlib import Path
import re

spec = importlib.util.spec_from_file_location(
    "source_score", Path(__file__).resolve().parent.parent / "source_action_pair_20261108/score.py")
source_score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source_score)


def state(text, arm):
    if arm != "report_with_decision":
        return source_score.opacity_state(text, arm)
    matches = list(re.finditer(r"airspace opacity or consolidation:\s*(yes|no|uncertain)\b", text.lower()))
    if len(matches) != 1 or text[matches[0].end():].strip().strip("."):
        return None
    return {"yes": 1, "no": 0, "uncertain": None}[matches[0][1]]


def summarize(results, reference):
    arms = ["describe", "focused", "report_with_decision"]
    by = {(x["index"], x["arm"]): x for x in results}
    metrics = {a: {"denominator": len(reference), "correct": 0, "unknown": 0, "completed": 0,
                   "negative_reference_correct": 0} for a in arms}
    correctness, scored = {}, []
    for item in reference:
        for arm in arms:
            row = by.get((item["index"], arm), {})
            pred = state(row.get("response", ""), arm) if row.get("status") == "completed" else None
            correct = pred is not None and pred == item["target_airspace_opacity"]
            correctness[item["index"], arm] = correct
            m = metrics[arm]
            m["correct"] += int(correct)
            m["unknown"] += int(pred is None)
            m["completed"] += int(row.get("status") == "completed")
            m["negative_reference_correct"] += int(correct and item["target_airspace_opacity"] == 0)
            scored.append({"index": item["index"], "arm": arm, "prediction": pred,
                           "reference": item["target_airspace_opacity"], "correct": correct})
    paired = {}
    for base in ["describe", "report_with_decision"]:
        paired[base + "_to_focused"] = {
            "gain": sum(not correctness[x["index"], base] and correctness[x["index"], "focused"] for x in reference),
            "loss": sum(correctness[x["index"], base] and not correctness[x["index"], "focused"] for x in reference)}
    p = paired["report_with_decision_to_focused"]
    complete = all(x["completed"] == len(reference) for x in metrics.values())
    return {"metrics": metrics, "paired": paired, "complete": complete,
            "candidate_gate_pass": complete and p["gain"] >= 2 and p["loss"] == 0 and metrics["focused"]["negative_reference_correct"] == 3,
            "report_concordance_proxy_only": True, "clinical_accuracy_established": False,
            "learning_established": False}, scored
