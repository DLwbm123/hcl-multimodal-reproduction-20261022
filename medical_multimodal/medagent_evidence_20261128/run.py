"""Train or score one source choice with identical read-only data and budget."""
from contextlib import nullcontext
import json
import os
from pathlib import Path
import random
import sys
import time

sys.path.append(str(Path(__file__).resolve().parent.parent / "ehr_plan_rl_20261128"))
import torch
from peft.utils.save_and_load import load_peft_weights

from evidence_task import ACTIONS, NAMES, prompt, score_policy, self_check, success
from train import CLIP, GROUP, KL, Policy, load_split, save

SEED = 26112892
STEPS = 64
BATCH = 4
LEARNING_RATE = 2e-5


class SourcePolicy(Policy):
    def __init__(self):
        super().__init__(ACTIONS)
        assert self.constrained

    def input_ids(self, case):
        text = self.tokenizer.apply_chat_template([{"role":"user","content":prompt(case)}],
                                                   tokenize=False, add_generation_prompt=True)
        return self.tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.cuda()


@torch.no_grad()
def evaluate(policy, rows, frozen=False):
    policy.model.eval()
    ctx = policy.model.disable_adapter() if frozen else nullcontext()
    with ctx:
        return score_policy(rows, lambda c: NAMES[int(policy.action_logprobs(policy.input_ids(c)).argmax())])


def objective(policy, row, mode, random_actions):
    ids = policy.input_ids(row)
    if mode == "sft":
        labels = [success(row, name) for name in NAMES]
        if not any(labels):
            return None, False, 0
        policy.model.train()
        logits = policy.action_logprobs(ids)
        return -logits[labels].mean(), len(set(labels)) > 1, 0
    policy.model.eval()
    with torch.no_grad(), policy.model.disable_adapter():
        reference = policy.action_logprobs(ids)
    policy.model.train()
    distribution = policy.action_logprobs(ids)
    sampled = torch.multinomial(distribution.detach().exp(), GROUP, replacement=True, generator=random_actions)
    # Only outcomes from the executed choices form on-policy advantages.
    reward = distribution.new_tensor([1.0 if success(row, NAMES[i]) else -1.0 for i in sampled.tolist()])
    advantage = reward - reward.mean()
    current = distribution[sampled]
    ratio = (current-current.detach()).exp()
    gain = torch.minimum(ratio*advantage, ratio.clamp(1-CLIP,1+CLIP)*advantage)
    reverse = reference[sampled]-current
    return (KL*(torch.expm1(reverse)-reverse)-gain).mean(), bool(advantage.abs().max()>0), GROUP


def fit(policy, train, dev, mode, out):
    rng = random.Random(SEED)
    order = list(range(len(train)))
    rng.shuffle(order)
    sampler = torch.Generator(device="cuda").manual_seed(SEED + 1)
    optimizer = torch.optim.AdamW(policy.trainable.values(), lr=LEARNING_RATE, weight_decay=0)
    original = policy.snapshot()
    best = original
    maximum = evaluate(policy,dev)[1]
    history = [dict(step=0,**maximum)]
    winner = updates = mixed = actions = labeled = 0
    for step in range(1,STEPS+1):
        optimizer.zero_grad(set_to_none=True)
        for i in range(BATCH):
            slot = ((step-1)*BATCH+i) % len(train)
            if slot==0 and step>1:
                rng.shuffle(order)
            loss,signal,draws = objective(policy, train[order[slot]], mode, sampler)
            if loss is not None:
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite objective")
                (loss/BATCH).backward()
                labeled += 1
            mixed += signal
            actions += draws
        norm = torch.nn.utils.clip_grad_norm_(policy.trainable.values(),1.,error_if_nonfinite=True)
        optimizer.step()
        updates += int(norm.item()>0)
        if step==1 or step%8==0:
            print(json.dumps(dict(step=step,updates=updates,signal=mixed,sampled_actions=actions)),flush=True)
        if step in (16,32,STEPS):
            metrics=evaluate(policy,dev)[1]
            history.append(dict(step=step,**metrics))
            if metrics["success"]>maximum["success"]:
                maximum=metrics
                best=policy.snapshot()
                winner=step
            save(out/"progress.json",dict(step=step,best_step=winner,dev=history))
    assert all(not p.requires_grad and p._version==version for p,version in policy.frozen)
    endpoint=policy.snapshot()
    changed=any(not torch.equal(v,original[k]) for k,v in endpoint.items())
    assert not updates or changed
    return best,endpoint,dict(best_step=winner,updates=updates,groups_with_signal=mixed,
                             sampled_actions=actions,loss_inputs=labeled,dev=history,adapter_changed=changed)


def main():
    self_check()
    gate = json.loads(Path("gate.json").read_text())
    assert gate["passes_training_gate"]
    mode = ("sft","rl")[int(os.environ["RUN_MODE"])]
    out = Path("a" if mode=="sft" else "b")
    out.mkdir(exist_ok=True)
    torch.manual_seed(SEED)
    policy = SourcePolicy()
    train,dev=(load_split(name) for name in ("train","dev"))
    before=evaluate(policy,dev[:8],frozen=True)[0]
    chosen,endpoint,stats=fit(policy,train,dev,mode,out)
    policy.restore(chosen)
    validation=evaluate(policy,dev[:8])[0]
    policy.model.save_pretrained(out/"adapter",save_embedding_layers=False)
    stored=load_peft_weights(out/"adapter",device="cpu")
    assert set(stored)==set(chosen) and all(torch.equal(v,stored[k]) for k,v in chosen.items())
    policy.restore(stored)
    assert evaluate(policy,dev[:8])[0]==validation and evaluate(policy,dev[:8],frozen=True)[0]==before
    policy.restore(endpoint)
    policy.model.save_pretrained(out/"endpoint",save_embedding_layers=False)
    (out/"selected").touch()
    other=Path("b" if mode=="sft" else "a")
    while not (other/"selected").exists():
        if (other/"stopped").exists():
            raise RuntimeError("paired arm did not select a checkpoint")
        time.sleep(2)
    test=load_split("test")
    predicted,metrics={},{}
    predicted["step64"],metrics["step64"]=evaluate(policy,test)
    if stats["best_step"]==STEPS:
        predicted["selected"],metrics["selected"]=predicted["step64"],metrics["step64"]
    else:
        policy.restore(stored)
        predicted["selected"],metrics["selected"]=evaluate(policy,test)
    if mode=="sft":
        predicted["frozen"],metrics["frozen"]=evaluate(policy,test,frozen=True)
    save(out/"private.json",predicted)
    save(out/"result.json",dict(mode=mode,test=metrics,training=stats,
                              max_reserved_mib=torch.cuda.max_memory_reserved()/2**20))


if __name__=="__main__":
    try:
        main()
    except Exception:
        failing=Path(("a","b")[int(os.environ["RUN_MODE"])])
        failing.mkdir(exist_ok=True)
        (failing/"stopped").touch()
        raise
