"""Matched GRPO ablation of the penalty for a successful unnecessary read."""
import math
import os
from pathlib import Path
import random
from statistics import mean
import sys
import time

here = Path(__file__).resolve().parent.parent
sys.path.extend(str(here / name) for name in ("ehr_plan_consistency_20261128", "ehr_plan_rl_20261128"))
import torch
from peft.utils.save_and_load import load_peft_weights

from environment import PLANS, execute
from panel import FORMS, view
from run import STATES, evaluate
from train import CLIP, GROUP, KL, Policy, SEED, audit_plans, load_split, save

STEPS = 384
EVAL_INTERVAL = 32
LEARNING_RATE = 2e-5
PENALTIES = (0.02, 0.20)


def reward(score, penalty):
    if score.success and score.unnecessary:
        return score.reward - (penalty - PENALTIES[0])
    return score.reward


def self_check():
    required = {"operation": "mean", "rows": [[1,80],[2,90]], "page": 1}
    sufficient = {**required, "page": 2}
    def samples(p):
        return [reward(execute(case, plan), p) for case, plan in (
            (required, ("mean",)), (required, ("more","mean")),
            (sufficient, ("mean",)), (sufficient, ("more","mean")),
            (sufficient, ("more","latest"))
        )]
    base, high = (samples(p) for p in PENALTIES)
    assert base == [-1, .96, .98, .96, -1]
    assert high[:3] == base[:3] and high[-1] == base[-1] and math.isclose(high[3], .78)


def rollout(policy, case, generator, penalty):
    prefix = policy.input_ids(case)
    policy.model.eval()
    with torch.no_grad(), policy.model.disable_adapter():
        reference = policy.action_logprobs(prefix)
    policy.model.train()
    distribution = policy.action_logprobs(prefix)
    indices = torch.multinomial(distribution.detach().exp(), GROUP, replacement=True, generator=generator)
    outcomes = [execute(case, PLANS[action]) for action in indices.tolist()]
    changed = [reward(score, penalty) for score in outcomes]
    rewards = distribution.new_tensor(changed)
    advantage = rewards - rewards.mean()
    current = distribution[indices]
    ratio = (current-current.detach()).exp()
    surrogate = torch.minimum(ratio*advantage, ratio.clamp(1-CLIP,1+CLIP)*advantage)
    reverse = reference[indices]-current
    loss = (KL*(torch.expm1(reverse)-reverse)-surrogate).mean()
    return loss, changed, [score.reward for score in outcomes], len(set(changed)) > 1


def checkpoint_key(metrics):
    counts = metrics["counts"]
    return counts["success"], -counts["unnecessary"]


def optimize(policy, patients, dev, penalty, out, *, seed=SEED, rank=checkpoint_key, capture=None):
    count = len(patients)
    assert count == 64 and STEPS % count == 0
    order = list(range(count))
    rng = random.Random(seed)
    forms = FORMS["train"]
    generator = torch.Generator(device="cuda").manual_seed(seed+1)
    optimizer = torch.optim.AdamW(policy.trainable.values(), lr=LEARNING_RATE, weight_decay=0.0)
    metrics, _ = evaluate(policy, dev, "dev")
    initial = policy.snapshot()
    best = initial
    best_key = rank(metrics)
    best_step = updates = mixed_groups = 0
    track = [dict(step=0, **metrics)]
    if capture is not None:
        capture(0)
    curve = []
    block_reward, block_common = [], []
    for step in range(1,STEPS+1):
        epoch, at = divmod(step-1, count)
        if at == 0:
            rng.shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        for state, slot in zip(STATES, (at, (at+count//2) % count)):
            first = (slot+epoch) % len(forms)
            second = (first+1+(slot//len(forms)) % (len(forms)-1)) % len(forms)
            for index in (first,second):
                case = view(patients[order[slot]], state, forms[index])
                loss, rewards, common, mixed = rollout(policy, case, generator, penalty)
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite policy objective")
                (loss/4).backward()
                block_reward.extend(rewards)
                block_common.extend(common)
                mixed_groups += int(mixed)
        norm = torch.nn.utils.clip_grad_norm_(policy.trainable.values(),1.0,error_if_nonfinite=True)
        optimizer.step()
        updates += int(norm.item()>0)
        if step==1 or step%16==0:
            print(dict(step=step, updates=updates, mixed_groups=mixed_groups),flush=True)
        if step%EVAL_INTERVAL==0:
            metrics, _ = evaluate(policy, dev, "dev")
            if rank(metrics)>best_key:
                best_key = rank(metrics)
                best = policy.snapshot()
                best_step = step
            curve.append(dict(step=step, mean_reward=mean(block_reward),
                              mean_common_reward=mean(block_common), sampled_actions=len(block_reward)))
            block_reward.clear()
            block_common.clear()
            track.append(dict(step=step,**metrics))
            if capture is not None:
                capture(step)
            save(out/"progress.json",dict(step=step,best_step=best_step,dev=track,curve=curve))
    assert all(not value.requires_grad and value._version == version for value,version in policy.frozen)
    endpoint = policy.snapshot()
    assert updates and any(not torch.equal(value,initial[name]) for name,value in endpoint.items())
    return best, endpoint, dict(best_step=best_step, dev=track, curve=curve, updates=updates,
                                mixed_groups=mixed_groups, sampled_actions=4*GROUP*STEPS)


def main():
    self_check()
    arm = int(os.environ["RUN_MODE"])
    out = Path(("a","b")[arm])
    out.mkdir(exist_ok=True)
    torch.manual_seed(SEED)
    policy = Policy()
    assert policy.constrained
    train,dev=(load_split(name) for name in ("train","dev"))
    fixtures=[view(x,STATES[0],FORMS["dev"][0]) for x in dev[:4]]
    frozen_before=audit_plans(policy,fixtures,frozen=True)
    start=time.monotonic()
    selected,endpoint,info=optimize(policy,train,dev,PENALTIES[arm],out)
    policy.restore(selected)
    predicted=audit_plans(policy,fixtures)
    policy.model.save_pretrained(out/"adapter",save_embedding_layers=False)
    weights=load_peft_weights(out/"adapter",device="cpu")
    assert set(weights)==set(selected) and all(torch.equal(v,selected[k]) for k,v in weights.items())
    policy.restore(weights)
    assert predicted==audit_plans(policy,fixtures) and frozen_before==audit_plans(policy,fixtures,frozen=True)
    policy.restore(endpoint)
    policy.model.save_pretrained(out/"endpoint",save_embedding_layers=False)
    (out/"selected").touch()
    other=Path(("b","a")[arm])
    while not (other/"selected").is_file():
        if (other/"stopped").is_file():
            raise RuntimeError("paired arm stopped before checkpoint selection")
        time.sleep(2)
    test=load_split("test")
    scored, traces = {}, {}
    scored["endpoint"],traces["endpoint"]=evaluate(policy,test,"test")
    if info["best_step"]==STEPS:
        scored["selected"],traces["selected"]=scored["endpoint"],traces["endpoint"]
    else:
        policy.restore(selected)
        scored["selected"],traces["selected"]=evaluate(policy,test,"test")
    if arm==0:
        for control in ("frozen","rule"):
            scored[control],traces[control]=evaluate(policy,test,"test",mode=control)
    save(out/"private.json",traces)
    save(out/"result.json",dict(penalty=PENALTIES[arm],training=info,test=scored,
                              seconds=round(time.monotonic()-start,1),
                              max_reserved_mib=round(torch.cuda.max_memory_reserved()/2**20,1)))


if __name__=="__main__":
    try:
        main()
    except Exception:
        failing=Path(("a","b")[int(os.environ["RUN_MODE"])])
        failing.mkdir(exist_ok=True)
        (failing/"stopped").touch()
        raise
