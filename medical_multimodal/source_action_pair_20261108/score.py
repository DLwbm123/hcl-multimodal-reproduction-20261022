"""Frozen conservative report-concordance proxy; never clinical adjudication."""
import re


def opacity_state(text, arm):
    text = text.strip().lower()
    if arm == "focused":
        # No inferred label from an unparsable/free-form or mixed yes/no answer.
        yes = bool(re.match(r"^yes\b", text))
        no = bool(re.match(r"^no\b", text))
        if yes and not re.search(r"\bno\b", text):
            return 1
        if no and not re.search(r"\byes\b", text):
            return 0
        return None
    target = r"\b(?:infiltrat\w*|opacit\w*|consolidat\w*|airspace shadowing)\b"
    positive, negative = False, False
    for sentence in re.split(r"[.!?;\n]", text):
        if re.search(target, sentence):
            if re.search(r"\b(no|without|absent|resolved)\b", sentence):
                negative = True
            else:
                positive = True
    normal = bool(re.search(r"\b(no abnormalities|no significant findings|normal chest|lungs are clear|clear lungs)\b", text))
    if positive and not negative and not normal:
        return 1
    if not positive and (negative or normal):
        return 0
    return None


def summarize(results, reference):
    by = {(r["index"], r["arm"]): r for r in results}
    metrics = {a: {"denominator": len(reference), "correct": 0, "unknown": 0, "completed": 0}
               for a in ["describe", "focused"]}
    groups, scored = {}, []
    for item in reference:
        group = groups.setdefault(item["source_group"], {"views": 0, "gain": 0, "loss": 0})
        group["views"] += 1
        correct = {}
        for arm in metrics:
            row = by.get((item["index"], arm), {})
            pred = opacity_state(row.get("response", ""), arm) if row.get("status") == "completed" else None
            correct[arm] = pred is not None and pred == item["target_airspace_opacity"]
            metrics[arm]["correct"] += int(correct[arm])
            metrics[arm]["unknown"] += int(pred is None)
            metrics[arm]["completed"] += int(row.get("status") == "completed")
            scored.append({"index": item["index"], "arm": arm, "prediction": pred,
                           "reference": item["target_airspace_opacity"], "correct": correct[arm]})
        group["gain"] += int(not correct["describe"] and correct["focused"])
        group["loss"] += int(correct["describe"] and not correct["focused"])
    gain, loss = sum(x["gain"] for x in groups.values()), sum(x["loss"] for x in groups.values())
    complete = all(m["completed"] == m["denominator"] for m in metrics.values())
    return {"metrics": metrics, "paired_gain": gain, "paired_loss": loss,
            "source_groups": len(groups), "group_counts": list(groups.values()),
            "complete": complete,
            "candidate_gate_pass": complete and gain >= 2 and loss == 0 and all(g["gain"] >= 1 for g in groups.values()),
            "metric_is_report_concordance_proxy": True, "clinical_accuracy_established": False,
            "learning_or_transfer_established": False}, scored
