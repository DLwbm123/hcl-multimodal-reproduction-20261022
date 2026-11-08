"""Run in the existing private runtime; emits counts, never image content.

MODEL_ROOT and TRACE_ROOT are explicit environment inputs. No weights are
loaded. The pinned tokenizer imports plotting helpers; reuse the complete
environment rather than installing or changing historical dependencies.
"""
import json
import os
from pathlib import Path


def main():
    from transformers import AutoTokenizer, dynamic_module_utils
    model = Path(os.environ["MODEL_ROOT"])
    root = Path(os.environ["TRACE_ROOT"])
    original = dynamic_module_utils.get_imports
    def imports(filename):
        result = original(filename)
        if Path(filename).name == "tokenization_chexagent.py":
            result = [s for s in result if s != "tensorflow"]
        return result
    try:
        dynamic_module_utils.get_imports = imports
        tokenizer = AutoTokenizer.from_pretrained(str(model), trust_remote_code=True, local_files_only=True)
    finally:
        dynamic_module_utils.get_imports = original
    context = json.loads((model / "config.json").read_text())["max_position_embeddings"]
    episodes = json.loads((root / "episodes.json").read_text())
    events = [json.loads(s) for s in (root / "events.jsonl").read_text().splitlines()]
    rows = []
    for i, event in enumerate(events):
        if event.get("kind") != "tool_started" or event["name"] != "chest_xray_expert":
            continue
        if episodes[event["episode"]]["state"] != "H0_train":
            continue
        args = event["args"]
        query = tokenizer.from_list_format([*({"image": p} for p in args["image_paths"]), {"text": args["prompt"]}])
        ids = tokenizer.apply_chat_template(
            [{"from": "system", "value": "You are a helpful assistant."}, {"from": "human", "value": query}],
            add_generation_prompt=True, return_tensors="pt")
        n, reserve = int(ids.shape[1]), args.get("max_new_tokens", 512)
        rows.append({"event_index": i, "episode": event["episode"], "images": len(args["image_paths"]),
                     "input_tokens": n, "max_new_tokens": reserve, "configured_context": context,
                     "input_exceeds_context": n > context, "reservation_exceeds_context": n + reserve > context})
    print(json.dumps({"rows": rows, "max_context": context, "tokenizer_only": True,
                      "image_decode_calls": 0, "model_inference_calls": 0}))


if __name__ == "__main__":
    main()
