"""Describe correctness changes conditional on observed operations, after scoring."""
import json
from collections import Counter
from pathlib import Path


def analyze(root):
    if not (root / "result.json").is_file():
        raise RuntimeError("Wait for sealed scoring results")
    pred = json.loads((root / "private_confirmation.json").read_text())["deployed"]
    summary = {}
    for stage, arms in pred.items():
        truth = json.loads((root / (stage+"_confirm") / "gold.json").read_text())
        for arm, rows in arms.items():
            counts = summary.setdefault(arm,Counter())
            for k, row in rows.items():
                ref = arms["base"][k]
                field = "trace" if "trace" in row else "views"
                different = ref[field] != row[field]
                group = "different_actions" if different else "same_actions"
                counts[group] += 1
                if ref["answer"] != truth[k] and row["answer"] == truth[k]:
                    counts[group+"_gain"] += 1
                if ref["answer"] == truth[k] and row["answer"] != truth[k]:
                    counts[group+"_loss"] += 1
                if row.get("actions") and any(a!="open_view" for a in row["actions"]):
                    counts["cases_with_crop_or_comparison"] += 1
    return summary


if __name__=="__main__":
    print(json.dumps(analyze(Path.cwd()),indent=2))
