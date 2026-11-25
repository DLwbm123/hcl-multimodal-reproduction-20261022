"""Paired acquisition study. The inference path reads only the prepared input manifest."""
import json
import os
import string
from pathlib import Path

from PIL import Image

ROOT = Path(os.environ["RUN_ROOT"])
DATA = Path(os.environ["DATA_ROOT"])
DESIGN = json.loads((ROOT / "design.json").read_text())


class ViewTool:
    def __init__(self, directory, case):
        self.directory = directory / "images" / case["case"]
        self.catalog = {item["view"]: item for item in case["views"]}
        self.opened = {}
        self.receipts = []

    def invoke(self, action):
        if action["name"] != "open_view":
            raise ValueError("unregistered tool")
        key = action["arguments"]["view_id"]
        if key not in self.catalog or key in self.opened or len(self.opened) >= 3:
            raise ValueError("invalid open_view argument or budget")
        path, = self.directory.glob(key + ".*")
        with Image.open(path) as source:
            self.opened[key] = source.convert("RGB")
        self.receipts.append({"action": action, "returned": path.stem, "size": list(self.opened[key].size)})


def open_action(view):
    return {"name": "open_view", "arguments": {"view_id": view}}


def load_vlm(*, with_text_generator=False):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    path = os.environ["MODEL_ROOT"]
    processor = AutoProcessor.from_pretrained(path, local_files_only=True,
        min_pixels=DESIGN["min_pixels"], max_pixels=DESIGN["max_pixels"])
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(path, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True, device_map={"": 0}).eval()
    model.requires_grad_(False)

    @torch.inference_mode()
    def choose(context, opened, instruction, labels):
        content = [{"type": "text", "text": context}]
        for key in sorted(opened):
            content.extend([{"type": "text", "text": "Opened image " + key}, {"type": "image"}])
        content.append({"type": "text", "text": instruction})
        prompt = processor.apply_chat_template([{"role": "user", "content": content}],
                                               tokenize=False, add_generation_prompt=True)
        batch = processor(text=[prompt], images=[opened[k] for k in sorted(opened)],
                          return_tensors="pt").to(model.device)
        token_ids = []
        for label in labels:
            token, = processor.tokenizer.encode(label, add_special_tokens=False)
            token_ids.append(token)
        # Only the next-token distribution is used; no text from other cases or prior calls.
        scores = model(**batch, use_cache=False).logits[0, -1, token_ids].float()
        if not torch.isfinite(scores).all():
            raise RuntimeError("nonfinite decision score")
        return labels[int(scores.argmax())]
    if with_text_generator:
        @torch.inference_mode()
        def generate(prompt, max_tokens):
            chat = [{"role": "user", "content": prompt}]
            text = processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], return_tensors="pt").to(model.device)
            tokens = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
            return processor.tokenizer.decode(tokens[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return choose, generate
    return choose


def episode(choose, directory, case, adaptive):
    tools = ViewTool(directory, case)
    history, options = case["history"], case["options"]
    catalog = list(tools.catalog)
    descriptions = [f"{v['view']} | {v['modality']} | {v['type']}" for v in case["views"]]
    context = ("Clinical history:\n" + history + "\nPossible diagnoses:\n" +
               "\n".join(f"{k}. {v}" for k,v in options.items()) +
               "\nStudy directory (modality and exam type only):\n" + "\n".join(descriptions))
    tools.invoke(open_action(catalog[0]))
    proposals = []
    for _ in range(2):
        remaining = [k for k in catalog if k not in tools.opened]
        if len(remaining) > len(string.ascii_uppercase):
            raise ValueError("too many candidate aliases")
        aliases = dict(zip(string.ascii_uppercase, remaining))
        instruction = ("Choose the next open_view call that would best distinguish the possible diagnoses "
                       "given the observed findings. Reply with the single action letter.\n" +
                       "\n".join(f"{a}. open_view({k})" for a,k in aliases.items()))
        action = open_action(aliases[choose(context, tools.opened, instruction, list(aliases))])
        proposals.append(action)
        tools.invoke(action if adaptive else open_action(remaining[0]))
    answer = choose(context, tools.opened, "Using the opened images and history, choose the most likely "
                    "diagnosis. Reply with its single option letter.", list(options))
    return {"answer": answer, "selector_actions": proposals, "receipts": tools.receipts,
            "selector_forwards": len(proposals), "diagnosis_forwards": 1}


def ids(arm):
    return [receipt["returned"] for receipt in arm["receipts"]]


def validate(row, catalog):
    fixed, adaptive = row["fixed"], row["adaptive"]
    for arm in [fixed,adaptive]:
        keys = ids(arm)
        if len(set(keys)) != 3 or len(keys) != 3 or keys[0] != catalog[0]:
            raise RuntimeError("view budget mismatch")
        if any(r["action"] != open_action(r["returned"]) for r in arm["receipts"]):
            raise RuntimeError("tool receipt mismatch")
    if ids(fixed) != catalog[:3] or adaptive["selector_actions"] != [r["action"] for r in adaptive["receipts"][1:]]:
        raise RuntimeError("policy mismatch")
    if set(ids(fixed)) == set(ids(adaptive)) and fixed["answer"] != adaptive["answer"]:
        raise RuntimeError("identical final inputs gave different diagnoses")


def main():
    choose = load_vlm()
    for split, expected in [("development",32), ("heldout",64)]:
        directory = DATA / split
        cases = [json.loads(s) for s in (directory / "inputs.jsonl").read_text().splitlines()]
        if len(cases) != expected or len({c["case"] for c in cases}) != expected:
            raise ValueError("panel mismatch")
        with (ROOT / ("private_" + split + ".jsonl")).open("x") as output:
            for i, case in enumerate(cases,1):
                row = {"case":case["case"]}
                for arm in ("fixed","adaptive"):
                    row[arm] = episode(choose, directory, case, arm == "adaptive")
                validate(row, [v["view"] for v in case["views"]])
                print(json.dumps(row),file=output,flush=True)
                if i == 1 or i % 8 == 0:
                    print(split,i,expected,flush=True)
    from score import summarize
    result = {s:summarize(ROOT, DATA, s) for s in ["development","heldout"]}
    (ROOT / "result.json").write_text(json.dumps(result,indent=2) + "\n")


if __name__ == "__main__":
    main()
