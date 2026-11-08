"""Paired visual-encoder instrumentation; no clinical target or optimizer."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

from token_budget import encode_checked
from cache_compat import inspect_prefill
from audit import response_contract


def main():
    r = Path(os.environ["DIAGNOSTIC_ROOT"])
    cfg = json.loads((r / "config.private.json").read_text())
    protocol = json.loads((r / "protocol.json").read_text())
    manifest = json.loads((r / "manifest.private.json").read_text())
    def save(name, value):
        p = r / name
        t = p.with_suffix(".tmp")
        t.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        t.replace(p)
    save("STARTED.json", {"time": time.time(), "pid": os.getpid(), "deadline": cfg["deadline"]})
    rows, fatal, generations = [], None, 0
    try:
        import torch
        import transformers
        from transformers import dynamic_module_utils
        from transformers.cache_utils import DynamicCache
        assert len(manifest) == 2 and all(x["privacy_approved"] for x in manifest)
        original_imports = dynamic_module_utils.get_imports
        def imports(filename):
            names = original_imports(filename)
            return [x for x in names if x != "tensorflow"] if Path(filename).name == "tokenization_chexagent.py" else names
        if not hasattr(DynamicCache, "seen_tokens"):
            DynamicCache.seen_tokens = property(lambda cache: cache.get_seq_length())
        if not hasattr(DynamicCache, "get_max_length"):
            DynamicCache.get_max_length = DynamicCache.get_max_cache_shape
        spec = importlib.util.spec_from_file_location("pinned_tool", cfg["upstream_tool"])
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        version = transformers.__version__
        try:
            dynamic_module_utils.get_imports = imports
            tool = module.XRayVQATool(model_name=cfg["model_root"], device="cuda:0", cache_dir=cfg["cache"])
        finally:
            dynamic_module_utils.get_imports, transformers.__version__ = original_imports, version
        context = tool.model.config.max_position_embeddings
        save("LOADED.json", {"time": time.time(), "transformers": version, "torch": torch.__version__, "context": context})
        for i, item in enumerate(manifest):
            for prompt_name, prompt in protocol["prompts"].items():
                variants = ["unmodified", "normalize_empty_prefill"]
                if i % 2:
                    variants.reverse()
                for variant in variants:
                    if time.time() >= cfg["deadline"]:
                        raise TimeoutError("Original diagnostic deadline reached")
                    start = time.time()
                    row = {"image_index": i, "prompt": prompt_name, "variant": variant, "status": "failed"}
                    try:
                        tokens, decision = encode_checked(tool.tokenizer, [item["path"]], prompt, protocol["max_new_tokens"], context)
                        row["token_admission"] = decision
                        if tokens is None:
                            row["status"] = "rejected_before_inference"
                        else:
                            with inspect_prefill(tool.model, variant == "normalize_empty_prefill") as observed:
                                generations += 1
                                result = tool._run([item["path"]], prompt, protocol["max_new_tokens"])
                            row.update(status=result[1].get("analysis_status", "unknown"), response=result[0].get("response"),
                                       observation=observed, output_contract=response_contract(str(result)))
                            if "error" in result[0]:
                                row["error"] = result[0]["error"]
                    except Exception as exc:
                        row["error"] = repr(exc)
                    row["seconds"] = time.time() - start
                    rows.append(row)
                    save("results.private.json", rows)
                    save("PROGRESS.json", {"time": time.time(), "recorded_cells": len(rows), "planned_cells": 8})
    except Exception as exc:
        fatal = repr(exc)
        save("failure.private.json", {"fatal": fatal, "traceback": traceback.format_exc()})
    seen = {(x["image_index"], x["prompt"], x["variant"]) for x in rows}
    for i in range(2):
        for prompt in protocol["prompts"]:
            for variant in ["unmodified", "normalize_empty_prefill"]:
                if (i, prompt, variant) not in seen:
                    rows.append({"image_index": i, "prompt": prompt, "variant": variant, "status": "not_run", "reason": fatal})
    save("results.private.json", rows)
    summary = {"finished_at": time.time(), "planned_cells": 8, "recorded_cells": len(rows),
               "completed_cells": sum(x["status"] == "completed" for x in rows), "generation_attempts": generations,
               "fatal": fatal, "clinical_correctness_established": False, "learning_gain_established": False}
    save("FINISHED.json", summary)
    return 0 if summary["completed_cells"] == 8 and fatal is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
