"""Evaluate an executor-side check for safe early termination of a mean query."""
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent / "ehr_plan_rl_20261128"))
import torch
from peft.utils.save_and_load import load_peft_weights

from environment import PLANS, execute, needs_more
from panel import variants
from train import Policy, load_split, save


def guard(plan, visible, has_more):
    # Unread rows are older than the last loaded row. Boundary timestamps are unique.
    if plan == ("mean",) and has_more and (not visible or visible[-1][0] < 24):
        return ("more", "mean")
    return plan


def evaluate_guard(policy, cases, arms):
    result = {}
    for label, adapter in arms:
        if adapter:
            policy.restore(load_peft_weights(adapter, device="cpu"))
        records = []
        wrong_operator = 0
        with torch.no_grad(), policy.model.disable_adapter() if adapter is None else nullcontext():
            for case in cases:
                raw = PLANS[int(policy.action_logprobs(policy.input_ids(case)).argmax())]
                wrong_operator += raw[-1] != case["operation"]
                page = case["rows"][:case["page"]]
                updated = guard(raw, page, len(page) < len(case["rows"]))
                records.append((execute(case, raw), execute(case, updated), needs_more(case), raw != updated))
        result[label] = dict(n=len(records), ordinary=sum(a.success for a, _, _, _ in records),
                             guarded=sum(b.success for _, b, _, _ in records),
                             rescued=sum(b.success and not a.success for a, b, _, _ in records),
                             degraded=sum(a.success and not b.success for a, b, _, _ in records),
                             guard_invoked=sum(d for _, _, _, d in records),
                             wrong_operator=wrong_operator,
                             early_stops=sum(a.premature for a, _, _, _ in records),
                             unnecessary_before=sum(a.unnecessary for a, _, _, _ in records),
                             unnecessary_after=sum(b.unnecessary for _, b, _, _ in records),
                             must_read_after=[sum(b.success and c for _, b, c, _ in records),
                                              sum(c for _, _, c, _ in records)])
    return result


def main():
    assert guard(("mean",), [[24, 1]], True) == ("mean",)
    assert guard(("mean",), [[23, 1]], True) == ("more", "mean")
    assert guard(("mean",), [[23, 1]], False) == ("mean",)
    assert guard(("latest",), [[1, 1]], True) == ("latest",)
    assert Path("a/selected").is_file() and Path("b/selected").is_file()
    policy = Policy()
    rows = load_split("test")
    arms = (("frozen", None), ("prior_rl", "i0/q1/a/adapter"),
            ("control", "a/adapter"), ("paired", "b/adapter"))
    if os.environ.get("RUN_OLD") == "1":
        old = json.loads(Path("i0/test.json").read_text())
        result = evaluate_guard(policy, old, (arms[1],))
        cases = [variants(case, i)[1] for i, case in enumerate(rows)]
        if Path("q2/b/adapter").is_dir():
            arms += (("paired_fixed", "q2/b/adapter"),)
        alternate = evaluate_guard(policy, cases, arms)
        save(Path("guard_other_wording.json"), alternate)
        save(Path("guard_old.json"), result)
        print(json.dumps(dict(old=result, alternate=alternate)))
    else:
        cases = [variants(case, i)[0] for i, case in enumerate(rows)]
        result = evaluate_guard(policy, cases, arms)
        save(Path("guard.json"), result)
        print(json.dumps(result))


if __name__ == "__main__":
    main()
