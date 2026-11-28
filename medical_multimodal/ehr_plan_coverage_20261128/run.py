"""Paired-page vs unchanged-page GRPO, matched patients, wording, and budget."""
from contextlib import nullcontext
import os
from pathlib import Path
from statistics import mean
import sys
import time

sys.path.append(str(Path(__file__).resolve().parent.parent / "ehr_plan_rl_20261128"))
import torch
from peft.utils.save_and_load import load_peft_weights

from environment import PLANS, execute, needs_more, rule
from panel import alternate_page, training_panels, variants
from train import Policy, SEED, audit_plans, evaluate, load_split, optimize, save


def coverage_diagnostic(policy, cases, mode):
    out = []
    context = policy.model.disable_adapter() if mode == "frozen" else nullcontext()
    with context, torch.no_grad():
        for i, row in enumerate(cases):
            options = variants(row, i)
            item = []
            for case in options:
                dist = policy.action_logprobs(policy.input_ids(case))
                plan = PLANS[int(dist.argmax())]
                item.append(dict(success=execute(case, plan).success,
                                 more=plan[0] == "more",
                                 probability=float(dist[2].exp()) if case["operation"] == "mean" else None))
            out.append(item)
    flips = []
    with policy.model.disable_adapter() if mode == "frozen" else nullcontext(), torch.no_grad():
        for i, row in enumerate(cases):
            alternate = alternate_page(row)
            if alternate is None:
                continue
            case = variants(row, i)[0]
            changed = {**case, "page": alternate}
            if not needs_more(case):
                case, changed = changed, case
            incomplete, complete = (policy.action_logprobs(policy.input_ids(r)) for r in (case, changed))
            correct = [PLANS[int(incomplete.argmax())] == rule(case),
                       PLANS[int(complete.argmax())] == rule(changed)]
            flips.append(dict(incomplete=correct[0], complete=correct[1], both=all(correct),
                              delta=float(incomplete[2].exp() - complete[2].exp())))
    summary = dict(n=len(cases),
                   wording_conflicts=sum(item[0]["success"] != item[1]["success"] for item in out),
                   new_wordings=[sum(item[i]["success"] for item in out) for i in range(2)],
                   page_pair=dict(n=len(flips), incomplete=sum(x["incomplete"] for x in flips),
                                  complete=sum(x["complete"] for x in flips),
                                  both=sum(x["both"] for x in flips),
                                  mean_p_more_shift=mean(x["delta"] for x in flips)))
    return summary


def main():
    mode = {"0": "control", "1": "paired"}[os.environ["RUN_MODE"]]
    out = Path("a" if mode == "control" else "b")
    out.mkdir(exist_ok=True)
    torch.manual_seed(SEED)
    policy = Policy()
    assert policy.constrained
    dev = [variants(case, i)[0] for i, case in enumerate(load_split("dev"))]
    train = training_panels(load_split("train"))[int(mode == "paired")]
    before = audit_plans(policy, dev[:4], frozen=True)
    best, stats = optimize(policy, train, dev, 64, "rl", out)
    fixed = os.environ.get("RUN_FIXED") == "1"
    if fixed:
        best = policy.snapshot()
    policy.restore(best)
    predictions = audit_plans(policy, dev[:4])
    policy.model.save_pretrained(out / "adapter", save_embedding_layers=False)
    written = load_peft_weights(out / "adapter", device="cpu")
    assert set(written) == set(best) and all(torch.equal(written[k], best[k]) for k in best)
    policy.restore(written)
    assert predictions == audit_plans(policy, dev[:4])
    assert before == audit_plans(policy, dev[:4], frozen=True)
    (out / "selected").touch()
    other = Path("b" if mode == "control" else "a")
    while not (other / "selected").is_file():
        if (other / "stopped").is_file():
            raise RuntimeError("paired arm did not complete checkpoint selection")
        time.sleep(2)
    rows = load_split("test")
    test = [variants(case, i)[0] for i, case in enumerate(rows)]
    panels = {"trained": "trainable"}
    if mode == "control" and not fixed:
        panels.update(frozen="frozen", rule="rule")
        comparison = load_peft_weights("i0/q1/a/adapter", device="cpu")
        panels["prior_rl"] = "trainable"
    summary, private = {}, {}
    for label, behavior in panels.items():
        if label == "prior_rl":
            policy.restore(comparison)
        result, scored = evaluate(policy, test, behavior)
        subgroup = [correct for case, correct in zip(test, scored) if needs_more(case)]
        result["must_read"] = [sum(subgroup), len(subgroup)]
        result["extra"] = coverage_diagnostic(policy, rows, behavior) if label != "rule" else None
        summary[label] = result
        private[label] = scored
    save(out / "private_scores.json", private)
    save(out / "result.json", dict(mode=mode, training=stats, test=summary, saved_weights_checked=True,
                                  evaluated_step=64 if fixed else stats["best_step"],
                                  visible_needs_in_training=sum(needs_more(case) for case in train),
                                  reserved_mib=round(torch.cuda.max_memory_reserved() / 2**20, 1)))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        out = Path("a" if os.environ["RUN_MODE"] == "0" else "b")
        out.mkdir(exist_ok=True)
        (out / "stopped").touch()
        raise
