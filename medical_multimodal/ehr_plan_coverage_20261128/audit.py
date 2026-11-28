"""Cross the old patients with every old question wording, preserving receipts."""
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from statistics import mean
import json
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent / "ehr_plan_rl_20261128"))
import torch
from peft.utils.save_and_load import load_peft_weights
from environment import PLANS, execute
from prepare import QUESTIONS
from train import Policy


def main():
    policy = Policy()
    roots = {"rl": "i0/q1/a/adapter", "sft": "i0/q1/b/adapter"}
    cohort = {}
    for split in ("dev", "test"):
        rows = json.loads(Path(f"i0/{split}.json").read_text())
        cohort[split] = [case for case in rows if case["stratum"] == "mean_more"]
    summary = {}
    traces = {}
    for mode in ("frozen", "sft", "rl"):
        if mode in roots:
            policy.restore(load_peft_weights(roots[mode], device="cpu"))
        with torch.no_grad(), policy.model.disable_adapter() if mode == "frozen" else nullcontext():
            cells = {}
            for split, rows in cohort.items():
                values = []
                for case in rows:
                    original = QUESTIONS["mean"].index(case["question"])
                    results = []
                    for text in QUESTIONS["mean"]:
                        probe = {**case, "question": text}
                        distribution = policy.action_logprobs(policy.input_ids(probe))
                        results.append(dict(success=execute(probe, PLANS[int(distribution.argmax())]).success,
                                            p_more_mean=float(distribution[2].exp()),
                                            margin=float(distribution[2] - distribution[0])))
                    values.append(dict(original=original, results=results, page=case["page"],
                                       failed_on_original=not results[original]["success"]))
                traces[(mode, split)] = values
                cells[split] = dict(
                    n=len(values),
                    page_hist=dict(Counter(c["page"] for c in values)),
                    original_pass=sum(c["results"][c["original"]]["success"] for c in values),
                    per_wording_pass=[sum(c["results"][i]["success"] for c in values) for i in range(3)],
                    per_wording_p_more=[round(mean(c["results"][i]["p_more_mean"] for c in values), 5) for i in range(3)],
                    original_p_more=round(mean(c["results"][c["original"]]["p_more_mean"] for c in values), 5)
                )
            summary[mode] = cells
    for mode in ("sft", "rl"):
        bad = [(before, after) for before, after in zip(traces[("frozen","test")], traces[(mode,"test")])
               if before["failed_on_original"]]
        summary[mode]["frozen_failed_test"] = dict(
            n=len(bad),
            original_p_more_before=round(mean(x["results"][x["original"]]["p_more_mean"] for x, _ in bad), 5),
            original_p_more_after=round(mean(x["results"][x["original"]]["p_more_mean"] for _, x in bad), 5),
            larger_margin=sum(y["results"][y["original"]]["margin"] > x["results"][x["original"]]["margin"] for x,y in bad),
            success_by_wording=[sum(y["results"][i]["success"] for _,y in bad) for i in range(3)]
        )
    result = Path("audit.json")
    result.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
