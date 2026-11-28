"""Train on-policy group-relative LoRA on EHR tool plans, or the paired SFT control."""
import argparse
from collections import Counter
from contextlib import nullcontext
import inspect
import json
import os
from pathlib import Path
import random
import re
from statistics import mean
import time
from unittest.mock import patch

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from peft.tuners.lora import model as lora_model
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

from environment import PLANS, encode, execute, parse, prompt, rule, self_check

SEED = 26112862
GROUP = 4
BATCH = 4
MAX_TOKENS = 64
KL = 0.02
CLIP = 0.2
LEARNING_RATE = 5e-6


def save(path, value):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False))
    pending.replace(path)


def load_split(split):
    return json.loads(Path(split + ".json").read_text())


class Policy:
    def __init__(self):
        path = os.environ["MODEL_ROOT"]
        self.constrained = os.environ.get("RUN_STYLE") == "1"
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True,
                                                       padding_side="left")
        self.base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            path, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
            local_files_only=True, device_map={"": 0}
        )
        self.base.requires_grad_(False)
        self.supports_logit_subset = "logits_to_keep" in inspect.signature(self.base.forward).parameters
        targets = [name for name, module in self.base.named_modules()
                   if isinstance(module, torch.nn.Linear)
                   and re.fullmatch(r"model\.(?:language_model\.)?layers\.\d+\.self_attn\.[qv]_proj", name)]
        if len(targets) != 2 * self.base.config.num_hidden_layers:
            raise RuntimeError(f"unable to identify both language attention projections per layer: {len(targets)}")
        lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
                          target_modules="(?:" + "|".join(map(re.escape, targets)) + ")")
        # PEFT 0.15.2 probes the unrelated local quantized backend even for BF16 Linear layers.
        with patch.object(lora_model, "is_bnb_available", return_value=False), \
                patch.object(lora_model, "is_bnb_4bit_available", return_value=False):
            self.model = get_peft_model(self.base, lora)
        self.trainable = {name: value for name, value in self.model.named_parameters() if value.requires_grad}
        assert len(self.trainable) == 2 * len(targets)
        assert all("lora_" in name and ".visual." not in name for name in self.trainable)
        self.frozen = [(value, value._version) for _, value in self.model.named_parameters()
                       if not value.requires_grad]
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.eval()
        config = self.model.generation_config
        self.eos = config.eos_token_id
        self.eos = (self.eos,) if isinstance(self.eos, int) else tuple(self.eos)
        self.pad = config.pad_token_id if config.pad_token_id is not None else self.eos[0]
        if self.constrained:
            sequences = [self.tokenizer.encode(encode(plan), add_special_tokens=False) + [self.eos[0]]
                         for plan in PLANS]
            width = max(map(len, sequences))
            self.plan_ids = torch.tensor([seq + [self.pad] * (width - len(seq)) for seq in sequences],
                                         device="cuda")
            self.plan_mask = torch.arange(width, device="cuda") < torch.tensor(
                [len(seq) for seq in sequences], device="cuda")[:, None]
            self.decision_tokens = []
            for seq in sequences:
                branches = []
                for step, token in enumerate(seq):
                    allowed = sorted({other[step] for other in sequences
                                      if len(other) > step and other[:step] == seq[:step]})
                    if len(allowed) > 1:
                        branches.append((step, torch.tensor(allowed, device="cuda"), allowed.index(token)))
                assert branches
                self.decision_tokens.append(branches)

    def input_ids(self, case):
        text = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt(case)}],
                                                  tokenize=False, add_generation_prompt=True)
        return self.tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.cuda()

    @torch.no_grad()
    def sample(self, prefix, group=1, stochastic=False):
        self.model.eval()
        if self.constrained:
            logprobs = self.action_logprobs(prefix)
            indices = torch.multinomial(logprobs.exp(), group, replacement=True) if stochastic else logprobs.argmax()[None]
            return self.plan_ids[indices], self.plan_mask[indices], [PLANS[i] for i in indices.tolist()]
        inputs = prefix.repeat(group, 1)
        ids = self.model.generate(
            input_ids=inputs, attention_mask=torch.ones_like(inputs),
            max_new_tokens=MAX_TOKENS, do_sample=stochastic, temperature=1.0,
            top_p=1.0, top_k=0, repetition_penalty=1.0, use_cache=True,
            eos_token_id=list(self.eos), pad_token_id=self.pad
        )[:, prefix.shape[1]:]
        stop = torch.isin(ids, torch.tensor(self.eos, device=ids.device))
        mask = stop.cumsum(dim=1) - stop.long() == 0
        plans = [parse(self.tokenizer.decode(seq[active], skip_special_tokens=True))
                 for seq, active in zip(ids, mask)]
        return ids, mask, plans

    def completion_logits(self, prefix, answers, mask):
        width = answers.shape[1]
        prefix = prefix.repeat(answers.shape[0], 1)
        tokens = torch.cat((prefix, answers), dim=1)
        attention = torch.cat((torch.ones_like(prefix), mask.long()), dim=1)
        subset = {"logits_to_keep": width + 1} if self.supports_logit_subset else {}
        result = self.model(input_ids=tokens, attention_mask=attention, use_cache=False, **subset)
        return result.logits[:, -width - 1:-1].float()

    def logprobs(self, prefix, answers, mask):
        logits = self.completion_logits(prefix, answers, mask)
        return -F.cross_entropy(logits.transpose(1, 2), answers, reduction="none")

    def action_logprobs(self, prefix):
        # Forced JSON tokens have probability one. Only grammar branches contribute to the action likelihood.
        logits = self.completion_logits(prefix, self.plan_ids, self.plan_mask)
        joint = torch.stack([
            torch.stack([logits[i, step, allowed].log_softmax(0)[choice]
                         for step, allowed, choice in branches]).sum()
            for i, branches in enumerate(self.decision_tokens)
        ])
        return joint.log_softmax(0)

    def snapshot(self):
        return {name: value.detach().cpu().clone()
                for name, value in get_peft_model_state_dict(self.model).items()}

    def restore(self, snapshot):
        set_peft_model_state_dict(self.model, snapshot)

    @torch.no_grad()
    def reference(self, prefix, nums, mask, warm_reference, indices):
        if warm_reference is None:
            with self.model.disable_adapter():
                return self.policy_logprobs(prefix, nums, mask, indices)
        current = self.snapshot()
        try:
            self.restore(warm_reference)
            return self.policy_logprobs(prefix, nums, mask, indices)
        finally:
            self.restore(current)

    def policy_logprobs(self, prefix, nums, mask, indices):
        if indices is not None:
            return self.action_logprobs(prefix)[indices][:, None]
        return self.logprobs(prefix, nums, mask)

    def supervised_loss(self, case, prefix):
        if self.constrained:
            return -self.action_logprobs(prefix)[PLANS.index(rule(case))]
        text = encode(rule(case))
        tokens = self.tokenizer.encode(text, add_special_tokens=False) + [self.eos[0]]
        ids = torch.tensor([tokens], device=prefix.device)
        mask = torch.ones_like(ids, dtype=torch.bool)
        return -self.logprobs(prefix, ids, mask).mean()


def diagnostic(policy, cases):
    grid = [[execute(case, action) for action in PLANS] for case in cases]
    sampled = []
    for case in cases:
        _, _, plans = policy.sample(policy.input_ids(case), GROUP, stochastic=True)
        sampled.append([execute(case, plan) for plan in plans])
    return dict(
        cases=len(cases), enumerated_reward_signal=sum(len({a.reward for a in row}) > 1 for row in grid),
        enumerated_success_signal=sum(len({a.success for a in row}) > 1 for row in grid),
        best_fixed_success=max(sum(row[i].success for row in grid) for i in range(len(PLANS))),
        rule_success=sum(execute(case, rule(case)).success for case in cases),
        oracle_success=sum(any(a.success for a in row) for row in grid),
        legal=sum(a.legal for row in sampled for a in row),
        successes=sum(a.success for row in sampled for a in row),
        draws=len(cases) * GROUP,
        mixed_success=sum(len({a.success for a in row}) > 1 for row in sampled),
        mixed_reward=sum(len({a.reward for a in row}) > 1 for row in sampled),
        mean_group_reward_variance=mean(mean((a.reward - mean(b.reward for b in row)) ** 2
                                             for a in row) for row in sampled)
    )


def evaluate(policy, cases, mode):
    rows = []
    for case in cases:
        if mode == "rule":
            question = case["question"].lower()
            operation = "mean" if re.search(r"\b(mean|average)\b", question) else "latest"
            plan = rule({**case, "operation": operation})
        else:
            with policy.model.disable_adapter() if mode == "frozen" else nullcontext():
                _, _, (plan,) = policy.sample(policy.input_ids(case))
        rows.append(execute(case, plan))
    counts = Counter(case["stratum"] for case in cases)
    correct = Counter(case["stratum"] for case, score in zip(cases, rows) if score.success)
    summary = dict(n=len(rows), success=sum(r.success for r in rows), legal=sum(r.legal for r in rows),
                   calls=sum(r.calls for r in rows), unnecessary=sum(r.unnecessary for r in rows),
                   premature=sum(r.premature for r in rows), reward=mean(r.reward for r in rows),
                   strata={key: [correct[key], n] for key, n in sorted(counts.items())})
    return summary, [r.success for r in rows]


def audit_plans(policy, cases, frozen=False):
    with policy.model.disable_adapter() if frozen else nullcontext():
        return [policy.sample(policy.input_ids(case))[0].tolist() for case in cases]


def paired(control, treated):
    rng = random.Random(SEED)
    changes = [int(new) - int(old) for old, new in zip(control, treated, strict=True)]
    samples = sorted(mean(rng.choices(changes, k=len(changes))) for _ in range(4000))
    return dict(gain=changes.count(1), loss=changes.count(-1),
                kept=sum(a and b for a, b in zip(control, treated)), delta=mean(changes),
                ci95=[samples[100], samples[3899]])


def optimize(policy, cases, dev, steps, mode, output, reference=None):
    rng = random.Random(SEED)
    order = list(range(len(cases)))
    rng.shuffle(order)
    optim = torch.optim.AdamW(policy.trainable.values(), lr=LEARNING_RATE, weight_decay=0.0)
    base = policy.snapshot()
    initial, _ = evaluate(policy, dev, "trainable")
    best = (initial["success"], initial["reward"])
    best_state = base
    best_step = 0
    history = [{"step": 0, **initial}]
    updates = 0
    sampled_groups = 0
    nonzero_groups = 0
    output_tokens = 0
    group_legal = 0
    group_success = 0
    adapter_changed = False
    checkpoints = {16, 32, steps}
    for step in range(1, steps + 1):
        optim.zero_grad(set_to_none=True)
        for index in range(BATCH):
            slot = ((step - 1) * BATCH + index) % len(cases)
            if not slot and step != 1:
                rng.shuffle(order)
            case = cases[order[slot]]
            prefix = policy.input_ids(case)
            if mode == "sft":
                output_tokens += len(policy.tokenizer.encode(encode(rule(case)), add_special_tokens=False)) + 1
                policy.model.train()
                loss = policy.supervised_loss(case, prefix)
            else:
                nums, mask, plans = policy.sample(prefix, GROUP, stochastic=True)
                output_tokens += int(mask.sum().item())
                outcomes = [execute(case, plan) for plan in plans]
                sampled_groups += 1
                group_legal += sum(outcome.legal for outcome in outcomes)
                group_success += sum(outcome.success for outcome in outcomes)
                rewards = torch.tensor([outcome.reward for outcome in outcomes], device=nums.device)
                advantage = (rewards - rewards.mean()).unsqueeze(1)
                nonzero_groups += int(bool(advantage.abs().max() > 1e-6))
                indices = torch.tensor([PLANS.index(plan) for plan in plans], device=nums.device) if policy.constrained else None
                active = torch.ones_like(advantage, dtype=torch.bool) if policy.constrained else mask
                with torch.no_grad():
                    old = policy.policy_logprobs(prefix, nums, mask, indices)
                    ref = policy.reference(prefix, nums, mask, reference, indices)
                policy.model.train()
                new = policy.policy_logprobs(prefix, nums, mask, indices)
                ratio = (new - old).exp()
                surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - CLIP, 1 + CLIP) * advantage)
                log_ratio = ref - new
                kl = torch.expm1(log_ratio) - log_ratio
                loss = (((KL * kl - surrogate) * active).sum(dim=1) / active.sum(dim=1)).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite optimization loss")
            (loss / BATCH).backward()

        norm = torch.nn.utils.clip_grad_norm_(policy.trainable.values(), 1.0, error_if_nonfinite=True)
        optim.step()
        updates += int(norm.item() > 0)
        if not adapter_changed and updates:
            adapter_changed = any(not torch.equal(weight, base[name]) for name, weight in policy.snapshot().items())
            assert adapter_changed
        if step == 1 or step % 8 == 0 or step == steps:
            print(json.dumps(dict(step=step, nonzero_updates=updates, nonzero_groups=nonzero_groups,
                                  output_tokens=output_tokens, allocated_mib=round(torch.cuda.max_memory_allocated()/2**20, 1))),
                  flush=True)
        if step in checkpoints:
            metrics, _ = evaluate(policy, dev, "trainable")
            history.append(dict(step=step, **metrics))
            key = (metrics["success"], metrics["reward"])
            if key > best:
                best = key
                best_step = step
                best_state = policy.snapshot()
            save(output / ("warm_progress.json" if mode == "sft" and steps == 16 else "progress.json"),
                 dict(history=history, best_step=best_step, nonzero_updates=updates))
    assert all(not weight.requires_grad and weight._version == version
               for weight, version in policy.frozen), "a frozen parameter changed"
    stats = dict(dev=history, best_step=best_step, updates=updates, adapter_changed=adapter_changed,
                 sampled_groups=sampled_groups, nonzero_groups=nonzero_groups,
                 sampled_completions=sampled_groups * GROUP, generated_tokens=output_tokens,
                 legal_rollouts=group_legal, successful_rollouts=group_success,
                 environment_executions=sampled_groups * GROUP)
    return best_state, stats


def run(mode, steps, out):
    self_check()
    torch.manual_seed(SEED)
    random.seed(SEED)
    dev = load_split("dev")
    cases = load_split("train")
    out.mkdir(exist_ok=True)
    policy = Policy()
    before = audit_plans(policy, dev[:4], frozen=True)
    if mode == "rl":
        initial = diagnostic(policy, dev)
        print(json.dumps({"preflight": initial}), flush=True)
        report = {"initial_preflight": initial}
        warm_reference = None
        if initial["legal"] / initial["draws"] < 0.8:
            warm_reference, report["warmup"] = optimize(policy, cases, dev, 16, "sft", out)
            policy.restore(warm_reference)
            initial = diagnostic(policy, dev)
            report["warm_preflight"] = initial
        if initial["legal"] / initial["draws"] < 0.8 or initial["mixed_reward"] == 0:
            report["status"] = "preflight_did_not_support_training"
            save(out / "result.json", report)
            (out / "stopped").touch()
            return
        report["optimization_objective"] = "success_and_efficiency" if initial["mixed_success"] else "efficiency_only"
        best, report["training"] = optimize(policy, cases, dev, steps, "rl", out, warm_reference)
        if warm_reference is not None:
            save(out / "warmup_audit.json", report["warmup"])
    else:
        report = {}
        best, report["training"] = optimize(policy, cases, dev, steps, "sft", out)

    policy.restore(best)
    adapter = out / "adapter"
    best_predictions = audit_plans(policy, dev[:4])
    policy.model.save_pretrained(adapter, save_embedding_layers=False)
    from peft.utils.save_and_load import load_peft_weights
    written = load_peft_weights(adapter, device="cpu")
    assert set(written) == set(best) and all(torch.equal(written[k], best[k]) for k in best)
    policy.restore(written)
    assert best_predictions == audit_plans(policy, dev[:4])
    assert before == audit_plans(policy, dev[:4], frozen=True)
    report["checks"] = dict(frozen_parameter_versions_unchanged=True, disabled_adapter_matches=True,
                            saved_adapter_matches=True, reloaded_dev_predictions_match=True)
    (out / "selected").touch()
    other = Path("b" if mode == "rl" else "a")
    while not (other / "selected").is_file():
        if (other / "stopped").is_file():
            report["status"] = "control_finished_no_rl_confirmation"
            save(out / "result.json", report)
            return
        time.sleep(2)
    picked, scores = evaluate(policy, load_split("test"), "trainable")
    report["test"] = {mode: picked}
    private = {mode: scores}
    if mode == "rl":
        for control in ("frozen", "rule"):
            metrics, reference_scores = evaluate(policy, load_split("test"), control)
            report["test"][control] = metrics
            report.setdefault("paired", {})[mode + "_vs_" + control] = paired(reference_scores, scores)
            private[control] = reference_scores
        if warm_reference is not None:
            policy.restore(warm_reference)
            metrics, reference_scores = evaluate(policy, load_split("test"), "trainable")
            report["test"]["warm_sft"] = metrics
            report["paired"]["rl_vs_warm_sft"] = paired(reference_scores, scores)
            private["warm_sft"] = reference_scores
    report["gpu"] = dict(allocated_mib=round(torch.cuda.max_memory_allocated()/2**20, 1),
                         reserved_mib=round(torch.cuda.max_memory_reserved()/2**20, 1))
    report["status"] = "complete"
    save(out / "private_scores.json", private)
    save(out / "result.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=64, choices=range(1, 65), metavar="1..64")
    args = parser.parse_args()
    mode = {"0": "rl", "1": "sft"}[os.environ["RUN_MODE"]]
    output = Path({"rl": "a", "sft": "b"}[mode])
    try:
        run(mode, args.steps, output)
    except Exception:
        output.mkdir(exist_ok=True)
        (output / "stopped").touch()
        raise
