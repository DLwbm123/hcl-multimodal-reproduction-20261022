"""Frozen visual/text model adapter used by both experiment workers."""
import json
import os
from pathlib import Path

ROOT = Path(os.environ["RUN_ROOT"])
DESIGN = json.loads((ROOT / "design.json").read_text())

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
        batch = processor(text=[prompt], images=[opened[k] for k in sorted(opened)] or None,
                          return_tensors="pt").to(model.device)
        token_ids = []
        for label in labels:
            token, = processor.tokenizer.encode(label, add_special_tokens=False)
            token_ids.append(token)
        # Only the next-token distribution is used; no text from other cases or prior calls.
        with (ROOT/'tokens.jsonl').open('a') as f:
            print(json.dumps(('choice', int(batch.input_ids.shape[1]), 0)), file=f)
        scores = model(**batch, use_cache=False).logits[0, -1, token_ids].float()
        if not torch.isfinite(scores).all():
            raise RuntimeError("nonfinite decision score")
        return labels[int(scores.argmax())]
    if with_text_generator:
        @torch.inference_mode()
        def generate(prompt, max_tokens, image=None):
            content = prompt if image is None else [{"type": "image"}, {"type": "text", "text": prompt}]
            chat = [{"role": "user", "content": content}]
            text = processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=None if image is None else [image], return_tensors="pt").to(model.device)
            tokens = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
            with (ROOT/'tokens.jsonl').open('a') as f:
                print(json.dumps(('generate', int(inputs.input_ids.shape[1]), int(tokens.shape[1] - inputs.input_ids.shape[1]))), file=f)
            return processor.tokenizer.decode(tokens[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return choose, generate
    return choose
