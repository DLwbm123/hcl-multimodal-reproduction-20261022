"""Two-arm source-only visual action diagnostic; references are scored locally later."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
from qualified_tool import QualifiedExpert


def main():
    r = Path(os.environ["DIAGNOSTIC_ROOT"])
    cfg = json.loads((r / "config.private.json").read_text())
    protocol = json.loads((r / "protocol.json").read_text())
    manifest = json.loads((r / "manifest.private.json").read_text())
    rows, events, fatal = [], [], None
    def save(name, value):
        p = r / name
        t = p.with_suffix(".tmp")
        t.write_text(json.dumps(value, indent=2) + "\n")
        t.replace(p)
    def record(kind, value):
        events.append({"kind": kind, **value})
        save("events.private.json", events)
    save("STARTED.json", {"time": time.time(), "pid": os.getpid()})
    try:
        import torch
        import transformers
        from transformers import dynamic_module_utils
        from transformers.cache_utils import DynamicCache
        assert len(manifest) == 7 and all(x["privacy_approved"] for x in manifest)
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
        state = {"images": set()}
        expert = QualifiedExpert(tool, state, record)
        for i, item in enumerate(manifest):
            state["images"] = {item["path"]}
            arms = list(protocol["arms"])
            if i % 2:
                arms.reverse()
            for arm in arms:
                if time.time() >= cfg["deadline"]:
                    raise TimeoutError("Original deadline reached")
                start = time.time()
                result = expert.invoke({"image_paths": [item["path"]],
                                        "prompt": protocol["arms"][arm],
                                        "max_new_tokens": protocol["max_new_tokens"]})
                rows.append({"index": item["index"], "arm": arm,
                             "status": result[1].get("analysis_status", "unknown"),
                             "response": result[0].get("response", ""),
                             "error": result[0].get("error"), "seconds": time.time() - start})
                save("results.private.json", rows)
                save("PROGRESS.json", {"recorded_cells": len(rows), "planned_cells": 14})
    except Exception as exc:
        fatal = repr(exc)
        save("failure.private.json", {"error": fatal, "traceback": traceback.format_exc()})
    seen = {(x["index"], x["arm"]) for x in rows}
    for item in manifest:
        for arm in protocol["arms"]:
            if (item["index"], arm) not in seen:
                rows.append({"index": item["index"], "arm": arm, "status": "not_run", "reason": fatal})
    save("results.private.json", rows)
    summary = {"finished_at": time.time(), "planned_cells": 14,
               "completed_cells": sum(x["status"] == "completed" for x in rows), "fatal": fatal,
               "generation_attempts": sum(x.get("generation_attempted", False) for x in events),
               "visual_calls": sum(x.get("visual_observation", {}).get("visual_calls", 0) for x in events),
               "controller_model_calls": 0, "source_groups": 2, "image_views": 7,
               "clinical_accuracy_evaluated": False, "learning_evaluated": False}
    save("FINISHED.json", summary)
    return 0 if summary["completed_cells"] == 14 and fatal is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
