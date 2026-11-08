"""Bounded tool qualification, not an Agent accuracy or learning experiment.

The private manifest contains only privacy-approved source images. Reuses pinned
weights and upstream tool. Run under an independent supervisor and deadline.
"""
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

from token_budget import encode_checked
from audit import response_contract


def main():
    root = Path(os.environ["DIAGNOSTIC_ROOT"])
    config = json.loads((root / "config.private.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    manifest = json.loads((root / "manifest.private.json").read_text())
    deadline = config["deadline"]
    assert len(manifest) == protocol["selected_images"] == 2
    assert all(x["privacy_approved"] and Path(x["path"]).is_file() for x in manifest)
    def save(name, value):
        target = root / name
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        tmp.replace(target)
    def check():
        if time.time() >= deadline:
            raise TimeoutError("Original diagnostic deadline reached")
    save("STARTED.json", {"time": time.time(), "deadline": deadline, "pid": os.getpid()})
    results, fatal, generated = [], None, 0
    try:
        import torch
        import transformers
        from transformers import dynamic_module_utils
        from transformers.cache_utils import DynamicCache
        model_root = Path(config["model_root"])
        max_context = json.loads((model_root / "config.json").read_text())["max_position_embeddings"]
        spec = importlib.util.spec_from_file_location("pinned_tool", config["upstream_tool"])
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        original_imports = dynamic_module_utils.get_imports
        def imports(filename):
            names = original_imports(filename)
            return [n for n in names if n != "tensorflow"] if Path(filename).name == "tokenization_chexagent.py" else names
        if not hasattr(DynamicCache, "seen_tokens"):
            DynamicCache.seen_tokens = property(lambda cache: cache.get_seq_length())
        if not hasattr(DynamicCache, "get_max_length"):
            DynamicCache.get_max_length = DynamicCache.get_max_cache_shape
        original_version = transformers.__version__
        try:
            dynamic_module_utils.get_imports = imports
            check()
            tool = module.XRayVQATool(model_name=str(model_root), device="cuda:0", cache_dir=config["cache"])
        finally:
            dynamic_module_utils.get_imports = original_imports
            transformers.__version__ = original_version
        save("LOADED.json", {"time": time.time(), "configured_context": max_context,
                             "transformers": transformers.__version__, "torch": torch.__version__})
        # Rotate order between the two images; no results are used for selection.
        for image_index, item in enumerate(manifest):
            arms = list(protocol["prompts"].items())
            if image_index % 2:
                arms.reverse()
            for arm, prompt in arms:
                check()
                start = time.time()
                row = {"image_index": image_index, "arm": arm, "status": "failed", "response": None}
                try:
                    tokens, decision = encode_checked(tool.tokenizer, [item["path"]], prompt,
                                                      protocol["max_new_tokens"], max_context)
                    row["token_admission"] = decision
                    if tokens is None:
                        row["status"] = "rejected_before_inference"
                    else:
                        generated += 1
                        result = tool._run([item["path"]], prompt, protocol["max_new_tokens"])
                        row.update(status=result[1].get("analysis_status", "unknown"),
                                   response=result[0].get("response"),
                                   output_contract=response_contract(str(result)))
                except Exception as exc:
                    row["error"] = repr(exc)
                row["seconds"] = time.time() - start
                results.append(row)
                save("results.private.json", results)
                save("PROGRESS.json", {"time": time.time(), "completed_cells": len(results), "planned_cells": 4})
    except Exception as exc:
        fatal = repr(exc)
        save("failure.private.json", {"fatal": fatal, "traceback": traceback.format_exc()})
    observed = {(x["image_index"], x["arm"]) for x in results}
    for i in range(2):
        for arm in protocol["prompts"]:
            if (i, arm) not in observed:
                results.append({"image_index": i, "arm": arm, "status": "not_run", "response": None, "reason": fatal})
    save("results.private.json", results)
    summary = {"finished_at": time.time(), "planned_cells": 4, "recorded_cells": len(results),
               "completed_cells": sum(x["status"] == "completed" for x in results),
               "generation_attempts": generated, "fatal": fatal,
               "clinical_correctness_established": False, "agent_accuracy_evaluated": False,
               "learned_policy_evaluated": False}
    save("FINISHED.json", summary)
    return 0 if not fatal and summary["completed_cells"] == 4 else 2


if __name__ == "__main__":
    raise SystemExit(main())
