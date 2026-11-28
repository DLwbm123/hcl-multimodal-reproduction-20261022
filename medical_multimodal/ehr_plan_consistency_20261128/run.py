"""Matched GRPO runs with an optional action-distribution consistency loss."""
from collections import Counter
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
from statistics import mean
import sys
import time

sys.path.append(str(Path(__file__).resolve().parent.parent / "ehr_plan_rl_20261128"))
import torch
from peft.utils.save_and_load import load_peft_weights

from environment import PLANS, execute
from evidence import Page, read_mean
from panel import FORMS, view
from train import CLIP, GROUP, KL, LEARNING_RATE, Policy, SEED, audit_plans, load_split, save

STATES = ("insufficient", "sufficient")
STEPS = 64
CHECKPOINTS = (16, 32, STEPS)
CONSISTENCY = 0.25


def divergence(left, right):
    mix = torch.logaddexp(left, right) - math.log(2)
    return ((left.exp() * (left - mix)).sum() + (right.exp() * (right - mix)).sum()) / 2


def grpo(policy, case, generator):
    ids = policy.input_ids(case)
    policy.model.eval()
    with torch.no_grad(), policy.model.disable_adapter():
        reference = policy.action_logprobs(ids)
    policy.model.train()
    distribution = policy.action_logprobs(ids)
    choices = torch.multinomial(distribution.detach().exp(), GROUP, replacement=True, generator=generator)
    actions = choices.tolist()
    outcomes = [execute(case, PLANS[index]) for index in actions]
    rewards = distribution.new_tensor([x.reward for x in outcomes])
    advantage = rewards - rewards.mean()
    current = distribution[choices]
    ratio = (current - current.detach()).exp()
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - CLIP, 1 + CLIP) * advantage)
    reverse_ratio = reference[choices] - current
    kl = torch.expm1(reverse_ratio) - reverse_ratio
    cost = int(policy.plan_mask[choices].sum().item())
    return distribution, (KL * kl - surrogate).mean(), len(set(rewards.tolist())) > 1, cost


def guard_mean(case):
    rows, size = case["rows"], case["page"]
    cursor = str(size) if size < len(rows) else None
    value = read_mean(Page(tuple(rows[:size]), cursor),
                      lambda _: Page(tuple(rows[size:]), None), unique_ages=True)
    full_window = [reading for age, reading in rows if age <= 24]
    expected = math.fsum(full_window) / len(full_window)
    assert value.value is not None and math.isclose(value.value, expected, abs_tol=1e-6, rel_tol=0)
    return value


@torch.no_grad()
def evaluate(policy, patients, split, mode="trained"):
    forms = FORMS[split]
    actions = {s: [Counter() for _ in forms] for s in STATES}
    patient_all = []
    patient_efficient = []
    consistencies = []
    costs = Counter()
    margins = []
    ctx = policy.model.disable_adapter() if mode == "frozen" else nullcontext()
    policy.model.eval()
    with ctx:
        for patient in patients:
            results, efficient, need_margins = [], [], []
            for state in STATES:
                distributions = []
                target = 2 if state == "insufficient" else 0
                first = view(patient, state, forms[0])
                checked = guard_mean(first)
                assert bool(checked.pages_read) == (state == "insufficient")
                for i, wording in enumerate(forms):
                    case = {**first, "question": wording}
                    if mode == "rule":
                        chosen = target
                    else:
                        lp = policy.action_logprobs(policy.input_ids(case))
                        distributions.append(lp)
                        chosen = int(lp.argmax())
                        if state == "insufficient":
                            other = torch.cat((lp[:target], lp[target+1:])).max()
                            need_margins.append(float(lp[target] - other))
                    action = PLANS[chosen]
                    scored = execute(case, action)
                    row = actions[state][i]
                    row.update(success=int(scored.success), efficient=int(chosen == target),
                               unnecessary=int(scored.unnecessary), premature=int(scored.premature),
                               tool_calls=scored.calls, guarded_success=int(scored.success or action == ("mean",)))
                    if action == ("mean",):
                        costs.update(extra_reads=checked.pages_read, added_rows=checked.added_in_window,
                                     confirmation_only=checked.confirmation_only_reads)
                    results.append(scored.success)
                    efficient.append(chosen == target)
                if distributions:
                    for i, left in enumerate(distributions):
                        for right in distributions[i+1:]:
                            consistencies.append(float(divergence(left, right)))
            patient_all.append(all(results))
            patient_efficient.append(all(efficient))
            if need_margins:
                margins.append(min(need_margins))
    totals = Counter({k: 0 for k in ("success", "efficient", "unnecessary", "premature", "tool_calls", "guarded_success")})
    for entries in actions.values():
        for row in entries:
            totals.update(row)
    summary = dict(patients=len(patients), prompts=len(patients) * len(forms) * 2,
                   all_success=sum(patient_all), all_efficient=sum(patient_efficient),
                   worst_required_form=min(row["success"] for row in actions["insufficient"]),
                   counts=dict(totals), cells={k: [dict(row) for row in v] for k, v in actions.items()},
                   mean_js=mean(consistencies) if consistencies else None,
                   mean_min_required_margin=mean(margins) if margins else None,
                   guard=dict(extra_reads=costs["extra_reads"], added_rows=costs["added_rows"],
                              confirmation_only=costs["confirmation_only"]))
    return summary, {"all_success": patient_all, "all_efficient": patient_efficient}


def checkpoint_key(metrics):
    return (metrics["all_efficient"], metrics["all_success"], metrics["worst_required_form"],
            -metrics["counts"]["unnecessary"])


def train(policy, patients, dev, weight, out):
    rng = random.Random(SEED)
    timeline = list(range(len(patients)))
    rng.shuffle(timeline)
    assert len(timeline) == STEPS
    forms = FORMS["train"]
    pairs = [(i % len(forms), (i + 1 + i // len(forms) % (len(forms) - 1)) % len(forms))
             for i in range(STEPS)]
    generator = torch.Generator(device="cuda").manual_seed(SEED + 1)
    optim = torch.optim.AdamW(policy.trainable.values(), lr=LEARNING_RATE, weight_decay=0.0)
    initial = policy.snapshot()
    best_state = initial
    metrics, _ = evaluate(policy, dev, "dev")
    best_key = checkpoint_key(metrics)
    best_step = 0
    history = [dict(step=0, **metrics)]
    rewards = Counter()
    costs = Counter()
    regularizers = []
    gradients = 0
    changed = False
    for step in range(1, STEPS + 1):
        optim.zero_grad(set_to_none=True)
        at = step - 1
        for state, slot in zip(STATES, (at, (at + STEPS // 2) % STEPS)):
            patient = patients[timeline[slot]]
            outputs = []
            for q in pairs[slot]:
                dist, loss, mixed, token_cost = grpo(policy, view(patient, state, forms[q]), generator)
                outputs.append((dist, loss))
                key = state + "_" + str(q)
                rewards[key] += int(mixed)
                costs[key] += 1
                costs["output_tokens"] += token_cost
            left, right = outputs
            js = divergence(left[0], right[0])
            regularizers.append(float(js.detach()))
            objective = (left[1] + right[1]) / 4 + weight * js / 2
            if not torch.isfinite(objective):
                raise RuntimeError("nonfinite policy objective")
            objective.backward()
        norm = torch.nn.utils.clip_grad_norm_(policy.trainable.values(), 1.0, error_if_nonfinite=True)
        optim.step()
        gradients += int(norm.item() > 0)
        if gradients and not changed:
            changed = any(not torch.equal(w, initial[k]) for k, w in policy.snapshot().items())
            assert changed
        if step == 1 or step % 8 == 0:
            print(json.dumps(dict(step=step, nonzero_updates=gradients, mixed_groups=sum(rewards.values()),
                                  sampled_actions=step * 4 * GROUP)), flush=True)
        if step in CHECKPOINTS:
            metrics, _ = evaluate(policy, dev, "dev")
            history.append(dict(step=step, **metrics))
            if checkpoint_key(metrics) > best_key:
                best_key = checkpoint_key(metrics)
                best_step = step
                best_state = policy.snapshot()
            save(out / "progress.json", dict(step=step, best_step=best_step, dev=history,
                                             nonzero_updates=gradients))
    assert all(not p.requires_grad and p._version == version for p, version in policy.frozen)
    by_cell = {k: [rewards[k], count] for k, count in costs.items() if k != "output_tokens"}
    assert sum(n for _, n in by_cell.values()) == 4 * STEPS
    return best_state, policy.snapshot(), dict(best_step=best_step, dev=history, updates=gradients,
        adapter_changed=changed, sampled_actions=4 * STEPS * GROUP, output_tokens=costs["output_tokens"],
        mixed_by_state_and_form=by_cell, average_training_js=mean(regularizers))


def main():
    mode = int(os.environ["RUN_MODE"])
    out = Path(("a", "b")[mode])
    out.mkdir(exist_ok=True)
    torch.manual_seed(SEED)
    train_rows, dev = (load_split(s) for s in ("train", "dev"))
    policy = Policy()
    assert policy.constrained
    audit = [view(x, STATES[0], FORMS["dev"][0]) for x in dev[:4]]
    before = audit_plans(policy, audit, frozen=True)
    start = time.monotonic()
    selected, endpoint, stats = train(policy, train_rows, dev, CONSISTENCY if mode else 0.0, out)
    policy.restore(selected)
    dir_selected = out / "adapter"
    prediction = audit_plans(policy, audit)
    policy.model.save_pretrained(dir_selected, save_embedding_layers=False)
    saved = load_peft_weights(dir_selected, device="cpu")
    assert set(selected) == set(saved) and all(torch.equal(v, saved[k]) for k, v in selected.items())
    policy.restore(saved)
    assert prediction == audit_plans(policy, audit) and before == audit_plans(policy, audit, frozen=True)
    policy.restore(endpoint)
    policy.model.save_pretrained(out / "endpoint", save_embedding_layers=False)
    (out / "selected").touch()
    other = Path(("b", "a")[mode])
    while not (other / "selected").is_file():
        if (other / "stopped").is_file():
            raise RuntimeError("the paired run did not select its checkpoint")
        time.sleep(2)
    test = load_split("test")
    summaries, scores = {}, {}
    summaries["step64"], scores["step64"] = evaluate(policy, test, "test")
    if stats["best_step"] == STEPS:
        summaries["selected"], scores["selected"] = summaries["step64"], scores["step64"]
    else:
        policy.restore(selected)
        summaries["selected"], scores["selected"] = evaluate(policy, test, "test")
    if mode == 0:
        for reference in ("frozen", "rule"):
            summaries[reference], scores[reference] = evaluate(policy, test, "test", mode=reference)
    save(out / "private.json", scores)
    save(out / "result.json", dict(test=summaries, training=stats, seconds=round(time.monotonic()-start, 1),
                                  max_reserved_mib=round(torch.cuda.max_memory_reserved()/2**20, 1),
                                  max_allocated_mib=round(torch.cuda.max_memory_allocated()/2**20, 1)))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        failing = Path(("a", "b")[int(os.environ["RUN_MODE"])])
        failing.mkdir(exist_ok=True)
        (failing / "stopped").touch()
        raise
