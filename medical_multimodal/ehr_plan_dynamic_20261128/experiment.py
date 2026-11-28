"""Three-stratum sampling with bounded rejection of equal-reward groups."""
from collections import Counter
from contextlib import nullcontext
import json
import os
from pathlib import Path
import random
from statistics import mean
import sys
import time

workspace = Path(__file__).resolve().parent.parent
sys.path.extend(str(workspace / name) for name in ("ehr_plan_consistency_20261128", "ehr_plan_rl_20261128"))
import torch
from peft.utils.save_and_load import load_peft_weights

from environment import PLANS, execute, prompt
from panel import FORMS, view
from run import STATES, guard_mean
from train import CLIP, KL, Policy, SEED, load_split, save

STRATA = ("required", "sufficient_with_next", "sufficient_last")
QUOTAS = dict(zip(STRATA, (1, 2, 1)))
GROUP = 8
STEPS = 384
ATTEMPT_CAP = 512
CANDIDATE_CAP = 48
BATCH = 8
INTERVAL = 32
LEARNING_RATE = 2e-5


def stratum(case):
    if case["state"] == "insufficient":
        return STRATA[0]
    return STRATA[1] if case["page"] < len(case["rows"]) else STRATA[2]


def reward(score):
    return score.reward - .18 if score.success and score.unnecessary else score.reward


class BatchPolicy(Policy):
    """Batch the existing complete, constrained JSON plans."""
    def __init__(self, expose_age):
        super().__init__()
        assert self.constrained
        self.expose_age = expose_age

    def case_tokens(self, case):
        text = prompt(case)
        if self.expose_age:
            visible = case["rows"][:case["page"]]
            age = visible[-1][0] if visible else None
            text += "\nAge of the oldest loaded observation in hours: " + json.dumps(age)
        chat = self.tokenizer.apply_chat_template([{"role":"user","content":text}],
                                                    tokenize=False, add_generation_prompt=True)
        return self.tokenizer.encode(chat, add_special_tokens=False)

    def batched(self, inputs):
        size = self.plan_ids.shape[0]
        width = max(map(len, inputs))
        device = self.model.device
        prefix = torch.tensor([[self.pad]*(width-len(s))+s for s in inputs],device=device)
        valid = torch.tensor([[0]*(width-len(s))+[1]*len(s) for s in inputs],device=device)
        ids = torch.cat((prefix.repeat_interleave(size,0), self.plan_ids.repeat(len(inputs),1)),dim=1)
        mask = torch.cat((valid.repeat_interleave(size,0), self.plan_mask.repeat(len(inputs),1)),dim=1)
        suffix = self.plan_ids.shape[1]
        option = {"logits_to_keep":suffix+1} if self.supports_logit_subset else {}
        logits = self.model(input_ids=ids,attention_mask=mask,use_cache=False,**option).logits
        logits = logits[:,-suffix-1:-1].reshape(len(inputs),size,suffix,-1)
        paths = [torch.stack([logits[:,i,pos,allowed].float().log_softmax(-1)[:,choice]
                              for pos,allowed,choice in path],dim=1).sum(-1)
                 for i,path in enumerate(self.decision_tokens)]
        return torch.stack(paths,dim=1).log_softmax(dim=-1)

    def scalar(self, tokens):
        return self.action_logprobs(torch.tensor([tokens],device=self.model.device))

    @torch.no_grad()
    def validate(self, tokenized):
        self.model.eval()
        fast = self.batched(tokenized)
        direct = torch.stack([self.scalar(tokens) for tokens in tokenized])
        single = torch.cat([self.batched([tokens]) for tokens in tokenized])
        torch.testing.assert_close(single,direct,atol=1e-5,rtol=1e-5)
        assert torch.isfinite(fast).all()
        return dict(max_abs_probability=float((fast.exp()-direct.exp()).abs().max()),
                    argmax_changes=int((fast.argmax(-1)!=direct.argmax(-1)).sum()), n=len(tokenized))


def cases_for(patients, split):
    return [dict(view(p,state,q), state=state)
            for p in patients for state in STATES for q in FORMS[split]]


def rows_for(policy, cases):
    return [policy.case_tokens(case) for case in cases]


@torch.no_grad()
def predict(policy, tokenized, base=False, exact=True):
    policy.model.eval()
    context = policy.model.disable_adapter() if base else nullcontext()
    with context:
        if exact:
            return torch.stack([policy.scalar(tokens) for tokens in tokenized])
        return torch.cat([policy.batched(tokenized[i:i+BATCH]) for i in range(0,len(tokenized),BATCH)])


def evaluate(policy, cases, tokenized, variants_per_patient, mode="trained"):
    if mode == "rule":
        choices = [2 if case["state"]=="insufficient" else 0 for case in cases]
    else:
        choices = predict(policy,tokenized,base=mode=="frozen").argmax(dim=1).tolist()
    cells={n:Counter({k:0 for k in ("n","success","unnecessary","premature","tool_calls","guarded_success",
                                     "guarded_premature","guarded_calls","extra_reads","efficient")}) for n in STRATA}
    successes, efficient, errors = [], [], []
    checks={}
    for i,(c,a) in enumerate(zip(cases,choices,strict=True)):
        state = stratum(c)
        s = execute(c,PLANS[a])
        ok = a == (2 if state==STRATA[0] else 0)
        if i%variants_per_patient==0:
            checks[i//variants_per_patient] = guard_mean(c)
        extra = checks[i//variants_per_patient].pages_read if a==0 else 0
        cells[state].update(n=1,success=int(s.success),unnecessary=int(s.unnecessary),premature=int(s.premature),
                            tool_calls=s.calls,guarded_success=int(s.success or a==0),
                            guarded_premature=int(s.premature and not extra),guarded_calls=s.calls+extra,
                            extra_reads=extra,efficient=int(ok))
        successes.append(s.success)
        efficient.append(ok)
        errors.append(a not in (0,2))
    totals = Counter()
    for counts in cells.values():
        totals.update(counts)
    per_patient = {"all_success":[all(successes[i:i+2*variants_per_patient])
                                  for i in range(0,len(cases),2*variants_per_patient)],
                   "all_efficient":[all(efficient[i:i+2*variants_per_patient])
                                   for i in range(0,len(cases),2*variants_per_patient)]}
    return dict(cells=cells,counts=totals,all_success=sum(per_patient["all_success"]),
                all_efficient=sum(per_patient["all_efficient"]),wrong_operator=sum(errors)), per_patient


def replenish(pool, targets, seen, rng, total):
    """Allocate proposals in proportion to the unfilled quota."""
    needs = [key for key,n in targets.items() for _ in range(n)]
    result = []
    for i in range(total):
        kind = needs[i%len(needs)]
        candidate = rng.choice([i for i in pool[kind] if i not in seen])
        result.append((kind,candidate))
        seen.add(candidate)
    return result


def collect(policy, pool, inputs, rewards, sampler, rng, counts):
    remaining = QUOTAS.copy()
    accepted = []
    seen=set()
    policy.model.eval()
    for start in range(0,CANDIDATE_CAP,BATCH):
        group = replenish(pool,remaining,seen,rng,min(BATCH,CANDIDATE_CAP-start))
        indices = [i for _,i in group]
        with torch.no_grad():
            lp=policy.batched([inputs[i] for i in indices])
            options=torch.multinomial(lp.exp(),GROUP,replacement=True,generator=sampler)
            outcomes=rewards[indices].gather(1,options)
            mixed=(outcomes.amax(dim=1)!=outcomes.amin(dim=1)).tolist()
        for j,(kind,i) in enumerate(group):
            counts[kind]["proposed"] += 1
            counts[kind]["mixed"] += mixed[j]
            if mixed[j] and remaining[kind]:
                remaining[kind] -= 1
                accepted.append((i,kind,options[j],outcomes[j],lp[j]))
        if not any(remaining.values()):
            return accepted
    return []


def update(policy, optimizer, group, inputs, reference):
    assert len(group)==4
    idx, actions, rewards, old = ([item[n] for item in group] for n in (0,2,3,4))
    actions, rewards, old = map(torch.stack, (actions,rewards,old))
    advantage=rewards-rewards.mean(dim=1,keepdim=True)
    optimizer.zero_grad(set_to_none=True)
    policy.model.train()
    current=policy.batched([inputs[i] for i in idx]).gather(1,actions)
    ratio=(current-old.gather(1,actions)).exp()
    posterior=reference[idx].gather(1,actions)-current
    objective=(KL*(torch.expm1(posterior)-posterior)-torch.minimum(
        ratio*advantage,ratio.clamp(1-CLIP,1+CLIP)*advantage)).mean()
    if not torch.isfinite(objective):
        raise RuntimeError("non-finite policy objective")
    objective.backward()
    norm=torch.nn.utils.clip_grad_norm_(policy.trainable.values(),1.0,error_if_nonfinite=True)
    optimizer.step()
    return float(norm)


def key(stats):
    return stats["counts"]["success"], -stats["counts"]["unnecessary"]


def train(policy, train_cases, dev, out, *, seed=SEED, rank=key, capture=None):
    tokens=rows_for(policy,train_cases)
    dev_tokens=rows_for(policy,dev)
    maximum_error=policy.validate(tokens[::63][:8])
    reference=predict(policy,tokens,base=True,exact=False)
    pool={s:[] for s in STRATA}
    verified_rewards=[]
    for i,c in enumerate(train_cases):
        pool[stratum(c)].append(i)
        verified_rewards.append([reward(execute(c,plan)) for plan in PLANS])
    assert all(len(p)>CANDIDATE_CAP for p in pool.values())
    rewards=torch.tensor(verified_rewards,device=policy.model.device)
    rng=random.Random(seed)
    sampler=torch.Generator(device="cuda").manual_seed(seed+1)
    optim=torch.optim.AdamW(policy.trainable.values(),lr=LEARNING_RATE,weight_decay=0)
    initial=policy.snapshot()
    chooser=initial
    best_step=0
    metrics,_=evaluate(policy,dev,dev_tokens,len(FORMS["dev"]))
    best=rank(metrics)
    attempts=steps=nonzero=0
    track=[dict(step=0,attempts=0,**metrics)]
    if capture is not None:
        capture(0)
    groups={kind:Counter({"proposed":0,"mixed":0,"used":0}) for kind in STRATA}
    blocks=[]
    while steps<STEPS and attempts<ATTEMPT_CAP:
        attempts+=1
        batch=collect(policy,pool,tokens,rewards,sampler,rng,groups)
        if batch:
            nonzero += int(update(policy,optim,batch,tokens,reference)>0)
            steps+=1
            for _,kind,*_ in batch:
                groups[kind]["used"]+=1
        if attempts%16==0 or attempts<=2:
            print(json.dumps(dict(attempts=attempts,step=steps,
                                  proposed=sum(n["proposed"] for n in groups.values()))),flush=True)
        due = batch and steps%INTERVAL==0
        if due or (attempts==ATTEMPT_CAP and steps!=track[-1]["step"]):
            metrics,_=evaluate(policy,dev,dev_tokens,len(FORMS["dev"]))
            if rank(metrics)>best:
                best=rank(metrics)
                best_step=steps
                chooser=policy.snapshot()
            track.append(dict(step=steps,attempts=attempts,**metrics))
            if capture is not None:
                capture(steps)
            blocks.append(dict(step=steps,attempts=attempts,by_state={k:dict(v) for k,v in groups.items()}))
            save(out/"progress.json",dict(step=steps,attempts=attempts,best_step=best_step,dev=track,blocks=blocks))
    assert all(not p.requires_grad and p._version==n for p,n in policy.frozen)
    end=policy.snapshot()
    updated=any(not torch.equal(initial[k],v) for k,v in end.items())
    assert not nonzero or updated
    return chooser,end,dict(best_step=best_step,steps=steps,attempts=attempts,nonzero=nonzero,
                            adapter_changed=updated,batch_audit=maximum_error,
                            dev=track,blocks=blocks,group_totals=groups,
                            sampled_actions=sum(c["proposed"] for c in groups.values())*GROUP)


def main():
    arm=int(os.environ["RUN_MODE"])
    out=Path("b" if arm else "a")
    out.mkdir(exist_ok=True)
    torch.manual_seed(SEED)
    policy=BatchPolicy(expose_age=bool(arm))
    a,b=(cases_for(load_split(s),s) for s in ("train","dev"))
    chosen,end,stats=train(policy,a,b,out)
    policy.restore(chosen)
    audit=rows_for(policy,b[:4])
    scores=predict(policy,audit)
    policy.model.save_pretrained(out/"adapter",save_embedding_layers=False)
    stored=load_peft_weights(out/"adapter",device="cpu")
    assert all(torch.equal(v,stored[k]) for k,v in chosen.items())
    policy.restore(stored)
    assert torch.equal(scores,predict(policy,audit))
    policy.restore(end)
    policy.model.save_pretrained(out/"endpoint",save_embedding_layers=False)
    (out/"selected").touch()
    other=Path("a" if arm else "b")
    while not (other/"selected").is_file():
        if (other/"stopped").is_file():
            raise RuntimeError("paired arm stopped")
        time.sleep(2)
    test=cases_for(load_split("test"),"test")
    tok=rows_for(policy,test)
    metrics,private={},{}
    metrics["endpoint"],private["endpoint"]=evaluate(policy,test,tok,len(FORMS["test"]))
    if stats["best_step"]==stats["steps"]:
        metrics["selected"],private["selected"]=metrics["endpoint"],private["endpoint"]
    else:
        policy.restore(stored)
        metrics["selected"],private["selected"]=evaluate(policy,test,tok,len(FORMS["test"]))
    if arm==0:
        for mode in ("frozen","rule"):
            metrics[mode],private[mode]=evaluate(policy,test,tok,len(FORMS["test"]),mode)
    else:
        metrics["frozen_with_age"],private["frozen_with_age"]=evaluate(policy,test,tok,len(FORMS["test"]),"frozen")
    save(out/"private.json",private)
    save(out/"result.json",dict(explicit_age=bool(arm),training=stats,test=metrics,
                              max_reserved_mib=torch.cuda.max_memory_reserved()/2**20))


if __name__=="__main__":
    try:
        main()
    except Exception:
        folder=Path(("a","b")[int(os.environ["RUN_MODE"])])
        folder.mkdir(exist_ok=True)
        (folder/"stopped").touch()
        raise
